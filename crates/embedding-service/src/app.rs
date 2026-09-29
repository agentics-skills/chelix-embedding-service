use std::{net::IpAddr, path::PathBuf, sync::Arc};

use {
    anyhow::{Context, Result, bail},
    chelix_embedding_service::api,
    tokio::signal,
};

use crate::engine::{LocalMistralEngine, ModelLoadArgs};

struct ServiceConfig {
    api_key: String,
    huggingface_api_key: String,
    model: String,
    cache_dir: PathBuf,
    listen_ip: IpAddr,
    listen_port: u16,
    rust_log: String,
}

fn load_service_config(env: impl Fn(&str) -> Option<String>) -> Result<ServiceConfig> {
    let api_key = required(&env, "API_KEY")?;
    let huggingface_api_key = required(&env, "HUGGINGFACE_API_KEY")?;
    let model = required(&env, "MODEL")?;
    let cache_dir = PathBuf::from(required(&env, "CACHE_DIR")?);
    let listen_ip = required(&env, "LISTEN_IP")?
        .parse::<IpAddr>()
        .context("LISTEN_IP is invalid")?;
    let listen_port = required(&env, "LISTEN_PORT")?
        .parse::<u16>()
        .context("LISTEN_PORT is invalid")?;
    if listen_port == 0 {
        bail!("LISTEN_PORT must not be 0");
    }
    let rust_log = required(&env, "RUST_LOG")?;
    Ok(ServiceConfig {
        api_key,
        huggingface_api_key,
        model,
        cache_dir,
        listen_ip,
        listen_port,
        rust_log,
    })
}

fn required(env: &impl Fn(&str) -> Option<String>, name: &str) -> Result<String> {
    match env(name) {
        Some(value) if !value.trim().is_empty() => Ok(value),
        Some(_) => bail!("{name} is empty"),
        None => bail!("{name} is not set"),
    }
}

pub(crate) async fn run() -> Result<()> {
    let config = load_service_config(|key| std::env::var(key).ok())?;
    init_tracing(&config.rust_log)?;
    let api_key = config.api_key.clone();
    let listen_ip = config.listen_ip;
    let listen_port = config.listen_port;
    let engine = LocalMistralEngine::new(ModelLoadArgs {
        model_spec: config.model,
        hf_token: config.huggingface_api_key,
        cache_dir: config.cache_dir,
    })
    .await?;
    let listener = tokio::net::TcpListener::bind((listen_ip, listen_port))
        .await
        .with_context(|| format!("binding embedding service on {listen_ip}:{listen_port}"))?;
    axum::serve(listener, api::router(Arc::new(engine), api_key))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("serving embedding API")
}

async fn shutdown_signal() {
    let ctrl_c = signal::ctrl_c();
    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            },
            Err(error) => {
                #[cfg(feature = "tracing")]
                tracing::error!(%error, "failed to install SIGTERM handler");
                #[cfg(not(feature = "tracing"))]
                let _ = error;
                std::future::pending::<()>().await;
            },
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        result = ctrl_c => {
            if let Err(error) = result {
                #[cfg(feature = "tracing")]
                tracing::error!(%error, "failed to wait for interrupt");
                #[cfg(not(feature = "tracing"))]
                let _ = error;
            }
        },
        () = terminate => {},
    }
}

#[cfg(feature = "tracing")]
fn init_tracing(rust_log: &str) -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_new(rust_log).context("RUST_LOG is invalid")?;
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stdout)
        .try_init()
        .map_err(|error| anyhow::anyhow!("initializing tracing: {error}"))?;
    Ok(())
}

#[cfg(not(feature = "tracing"))]
fn init_tracing(rust_log: &str) -> Result<()> {
    if rust_log.trim().is_empty() {
        bail!("RUST_LOG is empty");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        move |key| {
            vars.iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| (*value).to_string())
        }
    }

    fn complete_env() -> Vec<(&'static str, &'static str)> {
        vec![
            ("API_KEY", "client-key"),
            ("HUGGINGFACE_API_KEY", "hf-key"),
            ("MODEL", "org/model"),
            ("CACHE_DIR", "/var/cache/embeddings"),
            ("LISTEN_IP", "127.0.0.1"),
            ("LISTEN_PORT", "8080"),
            ("RUST_LOG", "info"),
        ]
    }

    #[test]
    fn missing_env_refuses_start() {
        let error = match load_service_config(env(&[])) {
            Err(error) => error,
            Ok(_) => panic!("empty env must fail"),
        };
        assert!(error.to_string().contains("is not set"), "{error}");
    }

    #[test]
    fn complete_env_parses_listen_address() {
        let config = load_service_config(env(&complete_env()))
            .unwrap_or_else(|error| panic!("config should parse: {error}"));
        assert_eq!(config.listen_ip.to_string(), "127.0.0.1");
        assert_eq!(config.listen_port, 8080);
        assert_eq!(config.model, "org/model");
        assert_eq!(config.api_key, "client-key");
    }
}

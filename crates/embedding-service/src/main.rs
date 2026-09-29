use std::io::Write;

mod app;
mod engine;

#[tokio::main]
async fn main() {
    if let Err(error) = app::run().await {
        let _ = writeln!(std::io::stdout(), "{error:#}");
        std::process::exit(1);
    }
}

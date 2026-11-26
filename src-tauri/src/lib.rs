use std::net::TcpListener;
use std::sync::Mutex;
use tauri::{Manager, RunEvent};
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

struct BackendState {
    url: String,
    child: Mutex<Option<CommandChild>>,
}

#[tauri::command]
fn get_backend_url(state: tauri::State<'_, BackendState>) -> String {
    state.url.clone()
}

fn find_available_port() -> u16 {
    if cfg!(debug_assertions) {
        return 8765;
    }
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|addr| addr.port())
        .unwrap_or(8765)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = find_available_port();
    let backend_url = format!("http://127.0.0.1:{port}");

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .manage(BackendState {
            url: backend_url.clone(),
            child: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![get_backend_url])
        .setup(move |app| {
            // In release builds, spawn the bundled Python AI engine sidecar automatically.
            // In dev mode, the developer runs `pnpm backend:dev` on port 8765.
            if !cfg!(debug_assertions) {
                if let Ok(sidecar) = app.shell().sidecar("voxloom-engine") {
                    if let Ok((_rx, child)) = sidecar
                        .args(["--host", "127.0.0.1", "--port", &port.to_string()])
                        .spawn()
                    {
                        let state = app.state::<BackendState>();
                        if let Ok(mut guard) = state.child.lock() {
                            *guard = Some(child);
                        }
                    }
                }
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building VoxLoom application");

    app.run(|app_handle, event| {
        if let RunEvent::Exit = event {
            let state = app_handle.state::<BackendState>();
            if let Ok(mut guard) = state.child.lock() {
                if let Some(child) = guard.take() {
                    let _ = child.kill();
                }
            }
        }
    });
}

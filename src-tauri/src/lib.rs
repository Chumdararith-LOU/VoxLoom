use std::net::TcpListener;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::{Manager, RunEvent};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

struct BackendState {
    url: String,
    child: Mutex<Option<Child>>,
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

fn spawn_bundled_engine(app: &tauri::App, port: u16) -> Option<Child> {
    let resource_dir = app.path().resource_dir().ok()?;
    let exe_name = if cfg!(target_os = "windows") {
        "voxloom-engine.exe"
    } else {
        "voxloom-engine"
    };

    let candidates = [
        resource_dir.join("engine").join(exe_name),
        resource_dir.join("resources").join("engine").join(exe_name),
    ];

    let engine_path = candidates.into_iter().find(|p| p.is_file())?;
    let mut cmd = Command::new(&engine_path);
    cmd.args(["--host", "127.0.0.1", "--port", &port.to_string()]);

    #[cfg(target_os = "windows")]
    cmd.creation_flags(CREATE_NO_WINDOW);

    cmd.spawn().ok()
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
            url: backend_url,
            child: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![get_backend_url])
        .setup(move |app| {
            // In release builds, spawn the bundled Python AI engine automatically.
            // In dev mode, run `pnpm backend:dev` on port 8765.
            if !cfg!(debug_assertions) {
                if let Some(child) = spawn_bundled_engine(app, port) {
                    let state = app.state::<BackendState>();
                    if let Ok(mut guard) = state.child.lock() {
                        *guard = Some(child);
                    };
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
                if let Some(mut child) = guard.take() {
                    let _ = child.kill();
                }
            };
        }
    });
}

# Rust Training Data
# Ownership, lifetimes, traits, async, error handling patterns.

```rust
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

// ═══════════════════════════════════════════════════════════════════════
// Ownership & Borrowing
// ═══════════════════════════════════════════════════════════════════════

/// A stack-allocated string buffer with fixed capacity.
struct FixedString {
    data: [u8; 256],
    len: usize,
}

impl FixedString {
    fn new() -> Self {
        FixedString {
            data: [0u8; 256],
            len: 0,
        }
    }

    fn push_str(&mut self, s: &str) {
        let bytes = s.as_bytes();
        let available = self.data.len() - self.len;
        let to_copy = bytes.len().min(available);
        self.data[self.len..self.len + to_copy].copy_from_slice(&bytes[..to_copy]);
        self.len += to_copy;
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.data[..self.len]).unwrap_or("")
    }
}

impl fmt::Display for FixedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Traits & Generics
// ═══════════════════════════════════════════════════════════════════════

trait Drawable {
    fn draw(&self, canvas: &mut Canvas);
    fn bounding_box(&self) -> Rect;
}

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Debug, Clone, Copy)]
struct Rect {
    min: Point,
    max: Point,
}

impl Rect {
    fn width(&self) -> f64 { self.max.x - self.min.x }
    fn height(&self) -> f64 { self.max.y - self.min.y }
    fn area(&self) -> f64 { self.width() * self.height() }
    fn contains(&self, p: Point) -> bool {
        p.x >= self.min.x && p.x <= self.max.x &&
        p.y >= self.min.y && p.y <= self.max.y
    }
}

struct Circle {
    center: Point,
    radius: f64,
}

struct Canvas {
    width: u32,
    height: u32,
    pixels: Vec<u32>,
}

impl Drawable for Circle {
    fn draw(&self, canvas: &mut Canvas) {
        // Midpoint circle algorithm (simplified)
        let cx = self.center.x as i32;
        let cy = self.center.y as i32;
        let r = self.radius as i32;

        let mut x = r;
        let mut y = 0;
        let mut err = 0;

        while x >= y {
            let points = [
                (cx + x, cy + y), (cx + y, cy + x),
                (cx - y, cy + x), (cx - x, cy + y),
                (cx - x, cy - y), (cx - y, cy - x),
                (cx + y, cy - x), (cx + x, cy - y),
            ];

            for (px, py) in points {
                if px >= 0 && px < canvas.width as i32 &&
                   py >= 0 && py < canvas.height as i32 {
                    canvas.pixels[(py * canvas.width as i32 + px) as usize] = 0xFFFFFF;
                }
            }

            y += 1;
            err += 1 + 2 * y;
            if 2 * (err - x) + 1 > 0 {
                x -= 1;
                err += 1 - 2 * x;
            }
        }
    }

    fn bounding_box(&self) -> Rect {
        Rect {
            min: Point { x: self.center.x - self.radius, y: self.center.y - self.radius },
            max: Point { x: self.center.x + self.radius, y: self.center.y + self.radius },
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Error Handling
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug)]
enum AppError {
    NotFound(String),
    Unauthorized,
    ValidationError(Vec<String>),
    DatabaseError(String),
    Internal(Box<dyn std::error::Error + Send + Sync>),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::NotFound(id) => write!(f, "Resource not found: {}", id),
            AppError::Unauthorized => write!(f, "Unauthorized access"),
            AppError::ValidationError(errors) => {
                write!(f, "Validation errors: {}", errors.join(", "))
            }
            AppError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            AppError::Internal(e) => write!(f, "Internal error: {}", e),
        }
    }
}

impl std::error::Error for AppError {}

type AppResult<T> = Result<T, AppError>;

// ═══════════════════════════════════════════════════════════════════════
// Iterators
// ═══════════════════════════════════════════════════════════════════════

struct FibonacciIterator {
    a: u64,
    b: u64,
}

impl FibonacciIterator {
    fn new() -> Self {
        FibonacciIterator { a: 0, b: 1 }
    }
}

impl Iterator for FibonacciIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.a;
        let new_b = self.a.checked_add(self.b)?;
        self.a = self.b;
        self.b = new_b;
        Some(result)
    }
}

// Collect first n fibonacci numbers that are even
fn even_fibs(n: usize) -> Vec<u64> {
    FibonacciIterator::new()
        .filter(|x| x % 2 == 0)
        .take(n)
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
// Concurrency
// ═══════════════════════════════════════════════════════════════════════

struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    sender: std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>,
}

impl ThreadPool {
    fn new(size: usize) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send>>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(size);

        for _ in 0..size {
            let rx = Arc::clone(&receiver);
            workers.push(thread::spawn(move || {
                loop {
                    let job = rx.lock().unwrap().recv();
                    match job {
                        Ok(task) => task(),
                        Err(_) => break,
                    }
                }
            }));
        }

        ThreadPool { workers, sender }
    }

    fn execute<F: FnOnce() + Send + 'static>(&self, f: F) {
        self.sender.send(Box::new(f)).unwrap();
    }
}

// Concurrent-safe cache
struct Cache<K, V> {
    data: RwLock<HashMap<K, V>>,
}

impl<K: std::hash::Hash + Eq, V: Clone> Cache<K, V> {
    fn new() -> Self {
        Cache {
            data: RwLock::new(HashMap::new()),
        }
    }

    fn get(&self, key: &K) -> Option<V> {
        self.data.read().unwrap().get(key).cloned()
    }

    fn set(&self, key: K, value: V) {
        self.data.write().unwrap().insert(key, value);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Builder Pattern
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug)]
struct HttpRequest {
    method: String,
    url: String,
    headers: HashMap<String, String>,
    body: Option<String>,
    timeout_ms: u64,
}

struct HttpRequestBuilder {
    method: String,
    url: String,
    headers: HashMap<String, String>,
    body: Option<String>,
    timeout_ms: u64,
}

impl HttpRequestBuilder {
    fn new(method: &str, url: &str) -> Self {
        HttpRequestBuilder {
            method: method.to_string(),
            url: url.to_string(),
            headers: HashMap::new(),
            body: None,
            timeout_ms: 30000,
        }
    }

    fn header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    fn body(mut self, body: String) -> Self {
        self.body = Some(body);
        self
    }

    fn timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    fn build(self) -> HttpRequest {
        HttpRequest {
            method: self.method,
            url: self.url,
            headers: self.headers,
            body: self.body,
            timeout_ms: self.timeout_ms,
        }
    }
}
```

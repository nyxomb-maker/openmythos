// TypeScript / JavaScript Training Data
// Modern web development patterns, async, types, React patterns.

// ═══════════════════════════════════════════════════════════════════════
// Type System
// ═══════════════════════════════════════════════════════════════════════

interface User {
  id: string;
  name: string;
  email: string;
  role: "admin" | "user" | "moderator";
  metadata?: Record<string, unknown>;
  createdAt: Date;
}

interface Post {
  id: string;
  title: string;
  content: string;
  author: User;
  tags: string[];
  publishedAt: Date | null;
  status: "draft" | "published" | "archived";
}

type UserCreateInput = Omit<User, "id" | "createdAt">;
type UserUpdateInput = Partial<Pick<User, "name" | "email" | "role">>;

// Generic Result type for error handling
type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

function ok<T>(data: T): Result<T> {
  return { success: true, data };
}

function err<E>(error: E): Result<never, E> {
  return { success: false, error };
}

// ═══════════════════════════════════════════════════════════════════════
// Async Patterns
// ═══════════════════════════════════════════════════════════════════════

async function fetchWithRetry<T>(
  url: string,
  options: RequestInit = {},
  maxRetries: number = 3,
  backoffMs: number = 1000
): Promise<Result<T>> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return ok(data as T);
    } catch (error) {
      if (attempt === maxRetries) {
        return err(error instanceof Error ? error : new Error(String(error)));
      }
      // Exponential backoff
      await new Promise((resolve) =>
        setTimeout(resolve, backoffMs * Math.pow(2, attempt))
      );
    }
  }
  return err(new Error("Unreachable"));
}

// Debounce with proper typing
function debounce<T extends (...args: any[]) => void>(
  func: T,
  waitMs: number
): T & { cancel: () => void } {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  const debounced = function (...args: Parameters<T>) {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), waitMs);
  } as T & { cancel: () => void };

  debounced.cancel = () => {
    if (timeoutId) clearTimeout(timeoutId);
  };

  return debounced;
}

// Throttle
function throttle<T extends (...args: any[]) => void>(
  func: T,
  limitMs: number
): T {
  let lastCall = 0;
  return function (...args: Parameters<T>) {
    const now = Date.now();
    if (now - lastCall >= limitMs) {
      lastCall = now;
      func(...args);
    }
  } as T;
}

// ═══════════════════════════════════════════════════════════════════════
// Event-Driven Architecture
// ═══════════════════════════════════════════════════════════════════════

type EventHandler<T = void> = (data: T) => void | Promise<void>;

class TypedEventEmitter<Events extends Record<string, any>> {
  private handlers = new Map<keyof Events, Set<EventHandler<any>>>();

  on<K extends keyof Events>(event: K, handler: EventHandler<Events[K]>): void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);
  }

  off<K extends keyof Events>(event: K, handler: EventHandler<Events[K]>): void {
    this.handlers.get(event)?.delete(handler);
  }

  async emit<K extends keyof Events>(event: K, data: Events[K]): Promise<void> {
    const handlers = this.handlers.get(event);
    if (!handlers) return;
    await Promise.all(
      Array.from(handlers).map((handler) => handler(data))
    );
  }
}

// Usage example
interface AppEvents {
  "user:login": { userId: string; timestamp: Date };
  "user:logout": { userId: string };
  "post:created": Post;
  "post:deleted": { postId: string };
}

const events = new TypedEventEmitter<AppEvents>();

// ═══════════════════════════════════════════════════════════════════════
// State Management (Redux-like)
// ═══════════════════════════════════════════════════════════════════════

type Action<T extends string = string, P = any> = {
  type: T;
  payload: P;
};

type Reducer<S, A extends Action> = (state: S, action: A) => S;

function createStore<S, A extends Action>(
  reducer: Reducer<S, A>,
  initialState: S
) {
  let state = initialState;
  const listeners = new Set<() => void>();

  return {
    getState: () => state,
    dispatch: (action: A) => {
      state = reducer(state, action);
      listeners.forEach((listener) => listener());
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

// ═══════════════════════════════════════════════════════════════════════
// React Component Patterns
// ═══════════════════════════════════════════════════════════════════════

/*
// Compound Components Pattern
function Tabs({ children, defaultTab }: { children: React.ReactNode; defaultTab: string }) {
  const [activeTab, setActiveTab] = useState(defaultTab);

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div className="tabs">{children}</div>
    </TabsContext.Provider>
  );
}

Tabs.Panel = function TabPanel({ id, children }: { id: string; children: React.ReactNode }) {
  const { activeTab } = useContext(TabsContext);
  if (activeTab !== id) return null;
  return <div className="tab-panel">{children}</div>;
};

// Custom Hook with cleanup
function useWebSocket(url: string) {
  const [messages, setMessages] = useState<string[]>([]);
  const [status, setStatus] = useState<"connecting" | "open" | "closed">("connecting");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setStatus("open");
    ws.onclose = () => setStatus("closed");
    ws.onmessage = (event) => {
      setMessages((prev) => [...prev, event.data]);
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [url]);

  const send = useCallback((message: string) => {
    wsRef.current?.send(message);
  }, []);

  return { messages, status, send };
}
*/

// ═══════════════════════════════════════════════════════════════════════
// Functional Programming Utilities
// ═══════════════════════════════════════════════════════════════════════

const pipe = <T>(...fns: Array<(arg: T) => T>) =>
  (value: T): T =>
    fns.reduce((acc, fn) => fn(acc), value);

const compose = <T>(...fns: Array<(arg: T) => T>) =>
  (value: T): T =>
    fns.reduceRight((acc, fn) => fn(acc), value);

// Option/Maybe monad
class Option<T> {
  private constructor(private readonly value: T | null) {}

  static some<T>(value: T): Option<T> {
    return new Option(value);
  }

  static none<T>(): Option<T> {
    return new Option<T>(null);
  }

  static from<T>(value: T | null | undefined): Option<T> {
    return value != null ? Option.some(value) : Option.none();
  }

  map<U>(fn: (value: T) => U): Option<U> {
    return this.value !== null ? Option.some(fn(this.value)) : Option.none();
  }

  flatMap<U>(fn: (value: T) => Option<U>): Option<U> {
    return this.value !== null ? fn(this.value) : Option.none();
  }

  getOrElse(defaultValue: T): T {
    return this.value !== null ? this.value : defaultValue;
  }

  match<U>(handlers: { some: (value: T) => U; none: () => U }): U {
    return this.value !== null ? handlers.some(this.value) : handlers.none();
  }
}

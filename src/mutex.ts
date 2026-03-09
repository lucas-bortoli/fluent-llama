/**
 * Helper class to create a promise with externally resolvable methods.
 * This allows other code to resolve the promise later without accessing
 * the internal promise constructor directly.
 *
 * @note This class is internal to the Mutex implementation.
 */
class ExternallyResolvable {
  public resolve: (() => void) | null;
  public readonly promise: Promise<void>;

  constructor() {
    this.resolve = null;
    this.promise = new Promise((resolve) => {
      this.resolve = resolve;
    });
  }
}

/**
 * An asynchronous Mutex implementation that serializes access to a critical section.
 *
 * This class ensures that only one async function can hold the lock at a time.
 * Waiters are queued in FIFO order.
 *
 * @example
 * ```typescript
 * const mutex = new Mutex();
 *
 * async function doWork() {
 *   // Acquire the lock
 *   const release = await mutex.lock();
 *   try {
 *     // Critical section
 *     await someAsyncOperation();
 *   } finally {
 *     // IMPORTANT: Release the lock in a finally block to ensure it is always called
 *     release();
 *   }
 * }
 * ```
 *
 * @warning This Mutex is **not re-entrant**.
 * Calling `lock()` from within a critical section that already holds the lock
 * will cause a deadlock. If you need re-entrancy, you must track the owner.
 *
 * @warning You **must** call the function returned by `lock()` to release the lock.
 * Forgetting to do so will cause the mutex to remain locked forever.
 */
export default class Mutex {
  private isLocked = false;
  private waitQueue: ExternallyResolvable[] = [];

  /**
   * Acquires the lock. Returns a function to release it.
   * @returns A function that releases the lock.
   */
  public async lock() {
    if (this.isLocked) {
      const myTurn = new ExternallyResolvable();
      this.waitQueue.push(myTurn);
      await myTurn.promise;
    }

    this.isLocked = true;

    // unlock function
    return () => {
      this.isLocked = false;

      const next = this.waitQueue.shift();
      next?.resolve?.();
    };
  }

  /**
   * Executes a critical section within the scope of the lock.
   * Automatically handles locking and unlocking to prevent leaks.
   *
   * @template T The expected return type of the critical section.
   * @param criticalSection - The async function to execute while holding the lock.
   * @returns A Promise resolving with the result of the critical section.
   *
   * @example
   * ```typescript
   * const result = await mutex.transaction(async () => {
   *   return await database.query("SELECT * ...");
   * });
   * ```
   */
  public async transaction(criticalSection: () => Promise<void>) {
    const release = await this.lock();
    try {
      await criticalSection();
    } finally {
      release();
    }
  }
}

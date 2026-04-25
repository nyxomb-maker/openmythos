"""
Example Python module — Data Structures & Algorithms
This file serves as training data for the LLM to learn Python syntax,
design patterns, and coding conventions.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import math


# ═══════════════════════════════════════════════════════════════════════
# Linked List
# ═══════════════════════════════════════════════════════════════════════

class ListNode:
    """Singly linked list node."""

    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

    def __repr__(self) -> str:
        vals = []
        node = self
        while node:
            vals.append(str(node.val))
            node = node.next
        return " -> ".join(vals)


def reverse_linked_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """Reverse a singly linked list in-place. O(n) time, O(1) space."""
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev


def merge_two_sorted_lists(
    l1: Optional[ListNode],
    l2: Optional[ListNode],
) -> Optional[ListNode]:
    """Merge two sorted linked lists into one sorted list."""
    dummy = ListNode()
    tail = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    tail.next = l1 if l1 else l2
    return dummy.next


def detect_cycle(head: Optional[ListNode]) -> bool:
    """Floyd's cycle detection. O(n) time, O(1) space."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════
# Binary Tree
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TreeNode:
    val: int = 0
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None


def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """Iterative in-order traversal using a stack."""
    result = []
    stack = []
    current = root

    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right

    return result


def max_depth(root: Optional[TreeNode]) -> int:
    """Maximum depth of a binary tree (recursive)."""
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """Check if a binary tree is a valid BST."""
    def validate(node, min_val, max_val):
        if not node:
            return True
        if node.val <= min_val or node.val >= max_val:
            return False
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    return validate(root, float('-inf'), float('inf'))


def lowest_common_ancestor(
    root: TreeNode, p: TreeNode, q: TreeNode
) -> TreeNode:
    """LCA of two nodes in a binary tree."""
    if not root or root == p or root == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root
    return left if left else right


# ═══════════════════════════════════════════════════════════════════════
# Graph Algorithms
# ═══════════════════════════════════════════════════════════════════════

class Graph:
    """Adjacency list graph representation."""

    def __init__(self, directed: bool = False):
        self.adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.directed = directed

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def bfs(self, start: int) -> List[int]:
        """Breadth-first search traversal."""
        visited = set()
        queue = deque([start])
        visited.add(start)
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor, _ in self.adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return order

    def dfs(self, start: int) -> List[int]:
        """Depth-first search traversal (iterative)."""
        visited = set()
        stack = [start]
        order = []

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            for neighbor, _ in reversed(self.adj[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

        return order

    def dijkstra(self, start: int) -> Dict[int, float]:
        """Dijkstra's shortest path algorithm."""
        dist = {start: 0.0}
        pq = [(0.0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            for v, w in self.adj[u]:
                new_dist = d + w
                if new_dist < dist.get(v, float('inf')):
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))

        return dist

    def topological_sort(self) -> List[int]:
        """Topological sort using Kahn's algorithm (BFS-based)."""
        in_degree = defaultdict(int)
        for u in self.adj:
            for v, _ in self.adj[u]:
                in_degree[v] += 1

        queue = deque([u for u in self.adj if in_degree[u] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor, _ in self.adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order


# ═══════════════════════════════════════════════════════════════════════
# Sorting Algorithms
# ═══════════════════════════════════════════════════════════════════════

def quicksort(arr: List[int]) -> List[int]:
    """Quicksort with random pivot — O(n log n) average."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def merge_sort(arr: List[int]) -> List[int]:
    """Merge sort — O(n log n) guaranteed."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left: List[int], right: List[int]) -> List[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# ═══════════════════════════════════════════════════════════════════════
# Dynamic Programming
# ═══════════════════════════════════════════════════════════════════════

def longest_common_subsequence(text1: str, text2: str) -> int:
    """LCS length using bottom-up DP. O(mn) time and space."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """0/1 Knapsack using DP. O(nW) time."""
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]


def coin_change(coins: List[int], amount: int) -> int:
    """Minimum number of coins to make amount. Returns -1 if impossible."""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1


# ═══════════════════════════════════════════════════════════════════════
# String Algorithms
# ═══════════════════════════════════════════════════════════════════════

def kmp_search(text: str, pattern: str) -> List[int]:
    """KMP string matching algorithm. O(n + m) time."""
    # Build failure function
    m = len(pattern)
    failure = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        failure[i] = j

    # Search
    matches = []
    j = 0
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = failure[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i - m + 1)
            j = failure[j - 1]

    return matches


def longest_palindromic_substring(s: str) -> str:
    """Expand-around-center approach. O(n^2) time, O(1) space."""
    if len(s) < 2:
        return s

    start, max_len = 0, 1

    def expand(left: int, right: int):
        nonlocal start, max_len
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if right - left + 1 > max_len:
                start = left
                max_len = right - left + 1
            left -= 1
            right += 1

    for i in range(len(s)):
        expand(i, i)       # Odd length
        expand(i, i + 1)   # Even length

    return s[start:start + max_len]


# ═══════════════════════════════════════════════════════════════════════
# Design Patterns — Observer
# ═══════════════════════════════════════════════════════════════════════

class EventEmitter:
    """Simple event emitter / observer pattern."""

    def __init__(self):
        self._listeners: Dict[str, List] = defaultdict(list)

    def on(self, event: str, callback):
        self._listeners[event].append(callback)

    def off(self, event: str, callback):
        self._listeners[event] = [
            cb for cb in self._listeners[event] if cb != callback
        ]

    def emit(self, event: str, *args, **kwargs):
        for callback in self._listeners.get(event, []):
            callback(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# Functional Programming Patterns
# ═══════════════════════════════════════════════════════════════════════

def compose(*functions):
    """Compose multiple functions right-to-left."""
    def composed(x):
        result = x
        for f in reversed(functions):
            result = f(result)
        return result
    return composed


def memoize(func):
    """Simple memoization decorator."""
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper


@memoize
def fibonacci(n: int) -> int:
    """Fibonacci with memoization. O(n) time."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

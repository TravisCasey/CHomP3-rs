// This file is part of CHomP3-rs, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! A vector-backed doubly linked list optimized for O(1) random removal.
//!
//! This module provides a specialized linked list implementation used
//! internally by [`CoreductionMatching`](super::CoreductionMatching) for
//! efficient manipulation of the Hasse diagram during the coreduction
//! algorithm.
//!
//! # Design Rationale
//!
//! The coreduction algorithm requires:
//! - O(1) append operations (`push_back`)
//! - O(1) removal by index (`pop`) - the key requirement
//! - O(1) front removal (`pop_front`)
//! - Cache-friendly memory layout
//!
//! Standard library [`std::collections::LinkedList`] does not support O(1)
//! removal by index (requires traversal). External crates like `slotmap` or
//! `slab` add dependency overhead for this specialized use case.
//!
//! This implementation stores nodes contiguously in a `Vec`, maintaining
//! doubly-linked pointers via indices. When `push_back` returns an index, that
//! index can later be used with `pop` for O(1) removal.

/// A node in the vector-backed doubly linked list.
///
/// Each node stores data and optional indices to its neighbors in the list.
/// When a node is removed, its parent/child pointers are cleared but the node
/// remains in the backing vector.
#[derive(Debug, Default, Clone)]
struct LLNode {
    /// The stored value.
    data: usize,
    /// Index of the previous node in the list, or `None` if this is the front.
    parent: Option<usize>,
    /// Index of the next node in the list, or `None` if this is the back.
    child: Option<usize>,
    /// Whether this node is currently active (i.e., not yet removed).
    active: bool,
}

/// A doubly linked list backed by a contiguous vector.
///
/// This data structure provides O(1) random access removal via the index
/// returned by [`push_back`](LinkedList::push_back), while maintaining linked
/// list semantics for sequential access.
#[derive(Debug, Default, Clone)]
pub(crate) struct LinkedList {
    nodes: Vec<LLNode>,
    front: Option<usize>,
    back: Option<usize>,
    size: usize,
}

impl LinkedList {
    /// Create a new empty linked list.
    #[must_use]
    pub fn new() -> Self {
        LinkedList {
            nodes: Vec::new(),
            front: None,
            back: None,
            size: 0,
        }
    }

    /// Create a new empty linked list with pre-allocated capacity.
    ///
    /// This is useful when the approximate number of elements is known in
    /// advance, avoiding reallocations during `push_back` operations.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        LinkedList {
            nodes: Vec::with_capacity(capacity),
            front: None,
            back: None,
            size: 0,
        }
    }

    /// Add an element to the back of the list.
    ///
    /// Returns the index of the newly inserted element. This index can be used
    /// later with [`pop`](LinkedList::pop) to remove the element in O(1) time.
    pub fn push_back(&mut self, data: usize) -> usize {
        let new_index = self.nodes.len();
        self.nodes.push(LLNode {
            data,
            parent: self.back,
            child: None,
            active: true,
        });

        // Update the previous back node to point to this new node
        if let Some(old_back) = self.back.replace(new_index) {
            self.nodes[old_back].child = Some(new_index);
        }

        // If this is the first node, also set front
        if self.front.is_none() {
            self.front = Some(new_index);
        }

        self.size += 1;
        new_index
    }

    /// Remove and return the element at the specified index.
    ///
    /// This is the key operation that provides O(1) random removal. The index
    /// should be one previously returned by
    /// [`push_back`](LinkedList::push_back).
    ///
    /// Returns `None` if the index is out of bounds or the element at that
    /// index has already been removed.
    pub fn pop(&mut self, index: usize) -> Option<usize> {
        if index >= self.nodes.len() || !self.nodes[index].active {
            return None;
        }

        let node = self.nodes[index].clone();

        // Update parent's child pointer
        if let Some(parent_index) = node.parent {
            self.nodes[parent_index].child = node.child;
        } else {
            // This was the front node
            self.front = node.child;
        }

        // Update child's parent pointer
        if let Some(child_index) = node.child {
            self.nodes[child_index].parent = node.parent;
        } else {
            // This was the back node
            self.back = node.parent;
        }

        // Mark the node as removed
        self.nodes[index].parent = None;
        self.nodes[index].child = None;
        self.nodes[index].active = false;

        self.size -= 1;
        Some(node.data)
    }

    /// Remove and return the front element of the list.
    ///
    /// Returns `None` if the list is empty.
    pub fn pop_front(&mut self) -> Option<usize> {
        self.front.map(|front_index| {
            debug_assert!(self.nodes[front_index].active, "front node is not active");
            let node = self.nodes[front_index].clone();

            // Update front pointer to the next node
            self.front = node.child;

            // Update child's parent pointer
            if let Some(child_index) = node.child {
                self.nodes[child_index].parent = None;
            } else {
                // This was the only node
                self.back = None;
            }

            // Mark the node as removed
            self.nodes[front_index].parent = None;
            self.nodes[front_index].child = None;
            self.nodes[front_index].active = false;

            self.size -= 1;
            node.data
        })
    }

    /// Returns `true` if the list contains no active elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the number of active elements in the list.
    ///
    /// This is an O(1) operation.
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns the data of the front element without removing it.
    ///
    /// Returns `None` if the list is empty.
    #[must_use]
    pub fn peek_front(&self) -> Option<usize> {
        self.front.map(|front_idx| self.nodes[front_idx].data)
    }

    /// Returns an iterator over the active elements in order.
    pub fn iter(&self) -> LinkedListIter<'_> {
        LinkedListIter {
            list: self,
            current: self.front,
        }
    }
}

/// An iterator over the active elements of a [`LinkedList`].
///
/// Created by [`LinkedList::iter`].
pub struct LinkedListIter<'a> {
    list: &'a LinkedList,
    current: Option<usize>,
}

impl Iterator for LinkedListIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.current.map(|current_index| {
            let node = &self.list.nodes[current_index];
            self.current = node.child;
            node.data
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iteration() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        let values: Vec<usize> = list.iter().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn iteration_after_removal() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.pop_front();

        let values: Vec<usize> = list.iter().collect();
        assert_eq!(values, vec![2, 3]);
    }

    #[test]
    fn peek_front() {
        let mut list = LinkedList::new();

        // Empty list
        assert_eq!(list.peek_front(), None);

        // Single element
        list.push_back(42);
        assert_eq!(list.peek_front(), Some(42));

        // Multiple elements - peek doesn't change
        list.push_back(100);
        assert_eq!(list.peek_front(), Some(42));

        // After pop_front, peek returns new front
        list.pop_front();
        assert_eq!(list.peek_front(), Some(100));

        // After emptying the list
        list.pop_front();
        assert_eq!(list.peek_front(), None);
    }

    #[test]
    fn push_back_and_linkage() {
        let mut list = LinkedList::new();
        let index1 = list.push_back(1);
        let index2 = list.push_back(2);
        let index3 = list.push_back(3);

        assert_eq!(index1, 0);
        assert_eq!(index2, 1);
        assert_eq!(index3, 2);
        assert_eq!(list.len(), 3);
        assert_eq!(list.front, Some(0));
        assert_eq!(list.back, Some(2));

        // Check linkage
        assert_eq!(list.nodes[0].child, Some(1));
        assert_eq!(list.nodes[1].parent, Some(0));
        assert_eq!(list.nodes[1].child, Some(2));
        assert_eq!(list.nodes[2].parent, Some(1));
    }

    #[test]
    fn pop_front_sequence() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.len(), 2);
        assert_eq!(list.front, Some(1)); // index 1 is now front

        assert_eq!(list.pop_front(), Some(2));
        assert_eq!(list.len(), 1);
        assert_eq!(list.front, Some(2)); // index 2 is now front

        assert_eq!(list.pop_front(), Some(3));
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());

        assert_eq!(list.pop_front(), None);
    }

    #[test]
    fn pop_by_index() {
        let mut list = LinkedList::new();
        let index1 = list.push_back(10);
        let index2 = list.push_back(20);
        let index3 = list.push_back(30);

        // Pop middle element
        assert_eq!(list.pop(index2), Some(20));
        assert_eq!(list.len(), 2);

        // Check that linkage is correct (first now points to third)
        assert_eq!(list.nodes[index1].child, Some(index3));
        assert_eq!(list.nodes[index3].parent, Some(index1));

        // Pop remaining elements
        assert_eq!(list.pop(index1), Some(10));
        assert_eq!(list.pop(index3), Some(30));
        assert!(list.is_empty());
    }

    #[test]
    fn pop_invalid_index() {
        let mut list = LinkedList::new();
        list.push_back(1);

        assert_eq!(list.pop(10), None);
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn with_capacity() {
        let list = LinkedList::with_capacity(100);
        assert!(list.is_empty());
        assert!(list.nodes.capacity() >= 100);
    }

    #[test]
    fn double_pop_returns_none() {
        let mut list = LinkedList::new();
        let idx = list.push_back(42);
        assert_eq!(list.pop(idx), Some(42));
        // Second pop on the same index must return None, not corrupt state
        assert_eq!(list.pop(idx), None);
        assert!(list.is_empty());
    }

    #[test]
    fn pop_front_then_pop_same_index() {
        let mut list = LinkedList::new();
        list.push_back(10);
        list.push_back(20);
        // pop_front removes index 0 (data=10)
        assert_eq!(list.pop_front(), Some(10));
        // Attempting to pop index 0 again returns None without corrupting list
        assert_eq!(list.pop(0), None);
        assert_eq!(list.len(), 1);
        // Remaining element (index 1, data=20) is still accessible
        assert_eq!(list.pop_front(), Some(20));
        assert!(list.is_empty());
    }

    #[test]
    fn pop_preserves_list_integrity() {
        let mut list = LinkedList::new();
        let i0 = list.push_back(1);
        let i1 = list.push_back(2);
        let i2 = list.push_back(3);

        // Remove middle, then try to double-remove it
        assert_eq!(list.pop(i1), Some(2));
        assert_eq!(list.pop(i1), None);

        // List should still iterate correctly over remaining elements
        let values: Vec<usize> = list.iter().collect();
        assert_eq!(values, vec![1, 3]);

        // Remove remaining in order
        assert_eq!(list.pop(i0), Some(1));
        assert_eq!(list.pop(i2), Some(3));
        assert!(list.is_empty());

        // Further pops return None, size stays 0
        assert_eq!(list.pop(i0), None);
        assert_eq!(list.pop(i2), None);
        assert_eq!(list.size, 0);
    }
}

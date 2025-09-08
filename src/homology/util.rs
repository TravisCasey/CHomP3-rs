// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/// A node in a vector-based doubly linked list.
#[derive(Debug, Default, Clone)]
struct LLNode {
    data: usize,
    parent: Option<usize>,
    child: Option<usize>,
}

/// A doubly linked list implemented using a vector to store nodes.
/// This provides O(1) random access to nodes via the [`LinkedList::pop`] method
/// while maintaining linked list semantics for insertions and deletions.
#[derive(Debug, Default, Clone)]
pub(crate) struct LinkedList {
    nodes: Vec<LLNode>,
    front: Option<usize>,
    back: Option<usize>,
    size: usize,
}

impl LinkedList {
    /// Create a new empty LinkedList.
    pub fn new() -> Self {
        LinkedList {
            nodes: Vec::new(),
            front: None,
            back: None,
            size: 0,
        }
    }

    /// Create a new empty LinkedList with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        LinkedList {
            nodes: Vec::with_capacity(capacity),
            front: None,
            back: None,
            size: 0,
        }
    }

    /// Add an element to the back of the list and return its index.
    pub fn push_back(&mut self, data: usize) -> usize {
        let new_index = self.nodes.len();
        self.nodes.push(LLNode {
            data,
            parent: self.back,
            child: None,
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
    /// Return `None` if the index is invalid or the node does not exist in the
    /// list.
    pub fn pop(&mut self, index: usize) -> Option<usize> {
        if index >= self.nodes.len() {
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

        // Mark the node as removed by clearing its pointers
        self.nodes[index].parent = None;
        self.nodes[index].child = None;

        self.size -= 1;
        Some(node.data)
    }

    /// Remove and return the front element of the list.
    /// Return `None` if the list is empty.
    pub fn pop_front(&mut self) -> Option<usize> {
        self.front.map(|front_index| {
            let node = self.nodes[front_index].clone();

            // Update front pointer to the next node
            self.front = node.child;

            // Update child's parent pointer
            if let Some(child_index) = node.child {
                self.nodes[child_index].parent = None;
            } else {
                // This was the back node
                self.back = None;
            }

            // Mark the node as removed by clearing its pointers
            self.nodes[front_index].parent = None;
            self.nodes[front_index].child = None;

            self.size -= 1;
            node.data
        })
    }

    /// Returns true if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the number of active nodes in the list.
    /// This is an O(1) operation.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns the data of the front element, if it exists.
    pub fn peek_front(&self) -> Option<usize> {
        self.front.map(|front_idx| self.nodes[front_idx].data)
    }

    /// Returns an iterator over the list elements.
    pub fn iter(&self) -> LinkedListIter<'_> {
        LinkedListIter {
            list: self,
            current: self.front,
        }
    }
}

/// An iterator over the elements of a LinkedList.
pub struct LinkedListIter<'a> {
    list: &'a LinkedList,
    current: Option<usize>,
}

impl<'a> Iterator for LinkedListIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current_index) = self.current {
            let node = &self.list.nodes[current_index];
            self.current = node.child;
            return Some(node.data);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        let values: Vec<usize> = list.iter().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_iter_empty() {
        let list = LinkedList::new();
        let values: Vec<usize> = list.iter().collect();
        assert_eq!(values, vec![]);
    }

    #[test]
    fn test_iter_after_operations() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        list.pop_front();

        let values: Vec<usize> = list.iter().collect();
        assert_eq!(values, vec![2, 3]);
    }

    #[test]
    fn test_peek_front() {
        let mut list = LinkedList::new();

        // Empty list
        assert_eq!(list.peek_front(), None);

        // Single element
        list.push_back(42);
        assert_eq!(list.peek_front(), Some(42));

        // Multiple elements
        list.push_back(100);
        assert_eq!(list.peek_front(), Some(42));

        // After pop_front
        list.pop_front();
        assert_eq!(list.peek_front(), Some(100));

        // After popping all elements
        list.pop_front();
        assert_eq!(list.peek_front(), None);
    }

    #[test]
    fn test_push_back_multiple_elements() {
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
    fn test_pop_front() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.len(), 2);
        assert_eq!(list.front, Some(1));

        assert_eq!(list.pop_front(), Some(2));
        assert_eq!(list.len(), 1);
        assert_eq!(list.front, Some(2));

        assert_eq!(list.pop_front(), Some(3));
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());

        assert_eq!(list.pop_front(), None);
    }

    #[test]
    fn test_pop_by_index() {
        let mut list = LinkedList::new();
        let index1 = list.push_back(10);
        let index2 = list.push_back(20);
        let index3 = list.push_back(30);

        // Pop middle element
        assert_eq!(list.pop(index2), Some(20));
        assert_eq!(list.len(), 2);

        // Check that linkage is correct
        assert_eq!(list.nodes[index1].child, Some(index3));
        assert_eq!(list.nodes[index3].parent, Some(index1));

        // Pop remaining elements
        assert_eq!(list.pop(index1), Some(10));
        assert_eq!(list.pop(index3), Some(30));
        assert!(list.is_empty());
    }

    #[test]
    fn test_pop_invalid_index() {
        let mut list = LinkedList::new();
        list.push_back(1);

        assert_eq!(list.pop(10), None);
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_with_capacity() {
        let list = LinkedList::with_capacity(100);
        assert!(list.is_empty());
        assert!(list.nodes.capacity() >= 100);
    }
}

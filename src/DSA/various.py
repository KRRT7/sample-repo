class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None  # Maintain a tail pointer

    def append(self, value):
        new_node = Node(value)
        if not self.head:
            self.head = new_node
            self.tail = new_node  # Initialize the tail
        else:
            self.tail.next = new_node  # Append at the tail
            self.tail = new_node  # Update the tail

    def find(self, value):
        current = self.head
        while current:
            if current.value == value:
                return True
            current = current.next
        return False


class Graph:
    def __init__(self):
        self.edges = []

    def add_edge(self, u, v):
        self.edges.append((u, v))

    def has_edge(self, u, v):
        return (u, v) in self.edges


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if self.stack:
            return self.stack.pop(0)
        return None

    def peek(self):
        if self.stack:
            return self.stack[0]
        return None

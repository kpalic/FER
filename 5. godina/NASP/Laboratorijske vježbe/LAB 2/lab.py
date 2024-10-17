from typing import Optional, Tuple, Callable
from enum import Enum


class AVLNode:
    """
    Class representing a single node of a AVL binary tree containing integer values.

    ...

    Attributes
    ----------

    value: int
        Value stored in the node.
    parent: Node, optional
        Parent of the current node. Can be None.
    left: Node, optional
        Left child of the current node. Can be None.
    right: Node, optional
        Right child of the current node. Can be None.
    height: int
        The height of the node (or the subtree rooted in this node).
    """
    def __init__(self, value: int) -> None:
        """
        Create a new AVLNode with the specified value, with its parent and children set to 0 and height set to 1.

        Args:
            value (int): The value contained in the new node.
        """
        self.value = value
        self.left = self.right = self.parent = None
        self.height = 1

    def set_left_child(self, node: Optional["AVLNode"]) -> None:
        """
        Set the the left child of self to the given node.
        Sets the node's parent to self (if it is not None).

        Args:
            node (Node, optional): the node to set as the child.
        """
        self.left = node
        if node is not None:
            node.parent = self

    def set_right_child(self, node: Optional["AVLNode"]) -> None:
        """
        Set the the right child of self to the given node.
        Sets the node's parent to self (if it is not None).

        Args:
            node (Node, optional): the node to set as the child.
        """
        self.right = node
        if node is not None:
            node.parent = self

    @staticmethod
    def node_height(node: Optional["AVLNode"]) -> int:
        """
        Helper static method to get the height of a node, but workaround None objects.

        Returns:
            int: The height of the node or 0 if the node is None
        """
        if node is None:
            return 0
        return node.height

    def _get_heights(self) -> Tuple[int, int]:
        """
        Helper method to fetch the heights of the left and right child.

        Returns:
            Tuple[int, int]: The heights of the left and right child as a tuple.
        """
        left_height = AVLNode.node_height(self.left)
        right_height = AVLNode.node_height(self.right)
        return left_height, right_height

    def update_height(self) -> None:
        """
        Update the height of this node by setting it to 1 + the maximum of the height of its (direct) child nodes.
        """
        self.height = 1 + max(*self._get_heights())

    def balance_factor(self) -> int:
        """
        Calculate the balance factor for this node (the subtree rooted in this node).

        Returns:
            int: The balance factor.
        """
        left_height, right_height = self._get_heights()
        return right_height - left_height

    def replace_child(self,
                      child: Optional['AVLNode'],
                      replacement: Optional['AVLNode']) -> None:
        """
        Replace the child of this node with the specified replacement node.

        If child is a reference to the left or the right child of this node it gets swapped with the
        specified replacement node.

        Args:
            child (AVLNode, optional): The child node to replace.
            replacement (AVLNode, optional): The node with which we want to replace the child.
        """
        if self.right is child:
            self.set_right_child(replacement)
        elif self.left is child:
            self.set_left_child(replacement)

    def __repr__(self) -> str:
        """
        Get the string representation of the node.

        Returns:
            str: A string representation which can create the AVLNode object.
        """
        return f'AVLNode({self.value})'


def _left_rotation_impl(rotatee: AVLNode, rotator: AVLNode) -> None:
    """
    Private static helper method for swapping the children of the rotatee and rotator.
    Specifically for left rotation.

    Args:
        rotatee (AVLNode, optional): The node around which we rotate (parent).

        rotator (AVLNode, optional): The node with which we rotate around the rotatee (child).
    """
    temp = rotator.left
    rotator.set_left_child(rotatee)
    rotatee.set_right_child(temp)

def _right_rotation_impl(rotatee: AVLNode, rotator: AVLNode) -> None:
    """
    Private static helper method for swapping the children of the rotatee and rotator.
    Specifically for right rotation.

    Args:
        rotatee (AVLNode, optional): The node around which we rotate (parent).
        rotator (AVLNode, optional): The node with which we rotate around the rotatee (child).
    """
    temp = rotator.right
    rotator.set_right_child(rotatee)
    rotatee.set_left_child(temp)

class RotationType(Enum):
    """
    Helper enum for types of rotation.
    The enum values are actually references to functions implementing rotations.
    """
    Left = _left_rotation_impl
    Right = _right_rotation_impl


class AVLTree:
    """
    Class representing an AVL self-balancing, binary tree containing integer values.

    ...

    Attributes
    ----------

    root: AVLNode, optional
        The root node of the tree.
    """
    def __init__(self, root: Optional[AVLNode] = None) -> None:
        """
        Create a AVLTree object with the specified root or None (by default).
        """
        self.root = root

    def set_root(self, root: Optional[AVLNode]) -> None:
        """
        Set the specified node as the root of this tree and remove its parent.

        Args:
            root (AVLNode, optional): The node which will be the new root of the tree.
        """
        self.root = root
        if self.root is not None:
            self.root.parent = None

    def rotate(self,
                    rotatee: Optional[AVLNode],
                    rotator: Optional[AVLNode],
                    rotate: RotationType) -> None:
        """
        Private helper method for implementing rotations.
        Accepts the rotatee and rotator Nodes which are used for the rotation and the rotation-specific
        'rotate' function which swaps the children of the rotatee and rotator as appropriate for the rotation.

        Args:
            rotatee (Node, optional): The node around which we rotate (parent).

            rotator (Node, optional): The node with which we rotate around the rotatee (child).

            rotate (RotationType): The type of rotation to perform. Left or right.
        """
        if rotatee is None or rotator is None or rotate is None:
            return
        parent = rotatee.parent
        if parent is None:
            self.set_root(rotator)
        else:
            if parent.left is rotatee:
                parent.set_left_child(rotator)
            else:
                parent.set_right_child(rotator)
        rotate(rotatee, rotator)


    
    def insert(self, value: int) -> bool:
        """
        Insert the value into the tree if it does not already exist (in it).

        Args:
            value (int): Value to insert into the tree.

        Returns:
            bool: True if the value was successfully inserted (was not alreay in the tree), False otherwise.
        """
        if self.root is None:
            self.root = AVLNode(value)
            return True

        node, parent = self.root, None
        attach_node_to_parent = lambda *args: None
        while node:
            if value == node.value:
                return False
            elif value < node.value:
                parent = node
                node = node.left
                attach_node_to_parent = parent.set_left_child
            elif value > node.value:
                parent = node
                node = node.right
                attach_node_to_parent = parent.set_right_child

        new_node = AVLNode(value)
        attach_node_to_parent(new_node)
        avl_balance(self, parent)

        return True

    def _get_predecessor(self, node: Optional[AVLNode]) -> Optional[AVLNode]:
        """
        Get the predecessor node of the specified node in the tree, if the predecessor exists.

        Args:
            node (AVLNode, optional): The node for which we want to find the predecessor

        Returns:
            AVLNode, optional: The predecessor node or None if it does not exist.
        """
        if node is None or node.left is None:
            return None
        parent, node = node.left, node.left.right
        while node is not None:
            parent = node
            node = node.right
        return parent
         
    def remove(self, value: int) -> bool:
        """
        Deletes the value specifed as the argument from the tree if it exists.
        If the value was not deleted (i.e. was not in the tree) returns False, otherwise True.

        Args:
            value (int): The value to delete from the tree.

        Returns:
            bool: True if the value was in the tree and was successfuly deleted, otherwise False.
        """
        if value is None or self.root is None:
            return False

        node = self.root
        
        while node is not None:
            if value < node.value:
                node = node.left
            elif value > node.value:
                node = node.right
            else:
                break

        if node is None:
            return False

        node_to_delete = self._get_predecessor(node)
        if node_to_delete is None:
            node_to_delete = node
        node.value = node_to_delete.value

        replacement = node_to_delete.right if node_to_delete.left is None else node_to_delete.left
        parent = node_to_delete.parent
        if parent is None:
            self.set_root(replacement)
        else:
            parent.replace_child(node_to_delete, replacement)
        avl_balance(self, parent)
        return True


def _rotate_and_update(tree: AVLTree, rotatee: AVLNode, rotator: AVLNode, rotation: RotationType) -> None:
    """
    Rotate the rotator node around the rotatee node in the specified tree, using the specified rotation implementation.
    Afterwards, update the heights of nodes where it is required.

    Args:
        tree (AVLTree): The tree in which the rotation takes place.

        rotatee (AVLNode): The node around which we rotate (parent in the rotation).

        rotator (AVLNode): The node which we rotate around the rotatee (the child in the rotation).

        rotation (RotationType): The type of rotation to perform. Can be RotationType.Left or RotationType.Right
    """
    tree.rotate(rotatee, rotator, rotation)
    
    rotatee.update_height()
    rotator.update_height()

def avl_balance(tree: AVLTree, node: Optional[AVLNode]) -> None:
    """
    Rebalance the tree starting from the node given as the argument and moving towards the root. 

    Args:
        node (AVLNode, optional): The node from which we start balancing.
    """
    
    # Iterate through the tree from the node to the root.
    # Update the heights and rebalance where needed
    while node is not None:
        node.update_height()
        balance_factor = node.balance_factor()
        
        if balance_factor < -1:
            if node.left and node.left.balance_factor() > 0:
                _rotate_and_update(tree, node.left, node.left.right, RotationType.Left)
            _rotate_and_update(tree, node, node.left, RotationType.Right)

        elif balance_factor > 1:
            if node.right and node.right.balance_factor() < 0:
                _rotate_and_update(tree, node.right, node.right.left, RotationType.Right)
            _rotate_and_update(tree, node, node.right, RotationType.Left)

        node = node.parent
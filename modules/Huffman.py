import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class Node(object):
    def __init__(self, label=None, value=0):
        self.label = label
        self.value = value
        self.left = None
        self.right = None


class HuffmanTree(object):
    def __init__(self, prob: dict):
        """
        :param prob: 256个整数的概率分布，神经网络的192个输出层对应192个HuffmanTree
        """
        self.prob = prob
        self.num_node = len(self.prob)
        self.lable = self.num_node  # new node label
        self.node_list = []
        self.encode_map = {}
        self.root = None
        self.buffer = ["0"] * 255
        self.sort_by_prob()
        self.init_tree()

    def sort_by_prob(self):
        """
        sort the prob by its probability
        """
        p = sorted(self.prob.items(), key=lambda i: i[1])
        self.prob = dict(p)

    def init_tree(self):
        """
        mainly init node_list
        """
        print("Initiating Nodes...")
        for item in self.prob.items():
            label, value = item
            node = Node(label=label, value=value)
            self.node_list.append(node)
        self.generate_tree()

    def find_node_by_label(self, label) -> Node:
        for node in self.node_list:
            if label == node.label:
                return node
        raise RuntimeError("Label not exists")

    def merge_node(self):
        """
        merge lowest 2 prob node
        """
        if len(self.prob) < 2:
            return False  # Not enough node to merge
        # update prob dict
        labels = list(self.prob.keys())
        left_label = labels[0]
        right_label = labels[1]
        left_node = self.find_node_by_label(left_label)
        right_node = self.find_node_by_label(right_label)
        left_value = left_node.value
        right_value = right_node.value
        new_value = left_value + right_value
        self.prob[self.lable] = new_value
        self.prob.pop(left_label)
        self.prob.pop(right_label)
        self.sort_by_prob()
        # add new node to HuffmanTree
        new_node = Node(label=self.lable, value=new_value)
        new_node.left = left_node
        new_node.right = right_node
        self.node_list.append(new_node)

        # update label
        self.lable += 1
        return True

    def generate_tree(self):
        print("Generating Tree...")
        while True:
            if not self.merge_node():
                break
        print("Huffman Tree is constructed successfully")
        self.root = self.find_node_by_label(list(self.prob.keys())[0])
        self.preorder_traversal(self.root)

    def preorder_traversal(self, root: Node, length=0):
        """
        preorder traversal and construct encode_map
        """
        if not root:
            return
        elif root.label < 256:
            self.encode_map[root.label] = "".join(self.buffer[:length])
        self.buffer[length] = '0'
        self.preorder_traversal(root.left, length + 1)
        self.buffer[length] = '1'
        self.preorder_traversal(root.right, length + 1)

    def encoder(self, img: torch.Tensor):
        """
        :param img: image to encode
        """
        img = img.numpy()
        pixels = list(img.flatten())
        code_pixels = list(map(lambda p: self.encode_map[p], pixels))
        return "".join(code_pixels)


if __name__ == '__main__':
    a = [[1, 1, 1, 1], [1, 1, 0, 0], [2, 2, 5, 1], [0, 0, 1, 0]]
    prob = {0: 0.3125, 1: 0.5, 2: 0.125, 5: 0.0625}
    h = HuffmanTree(prob)
    img = torch.tensor(a)
    code = h.encoder(img)
    print("优化位数:", 2, len(code) / (img.size()[0] * img.size()[1]))
    print(code)

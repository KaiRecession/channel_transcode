# -*- coding: UTF-8 -*-
import numpy as np

from generate_hlsvideo import data_load


class DoubleLinkedList:

    def __init__(self, capacity=0xffffffff):
        """
        双向链表
        :param capacity: 链表容量 初始化为int的最大值2^32-1
        :return:
        """
        self.capacity = capacity
        self.size = 0
        self.head = None
        self.tail = None

    def __add_head(self, node):
        """
        向链表头部添加节点
            头部节点不存在 新添加节点为头部和尾部节点
            头部节点已存在 新添加的节点为新的头部节点
        :param node: 要添加的节点
        :return: 已添加的节点
        """
        # 头部节点为空
        if not self.head:
            self.head = node
            self.tail = node
            self.head.next = None
            self.tail.prev = None
        # 头部节点不为空
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
            self.head.prev = None
        self.size += 1

        return node

    def __add_tail(self, node):
        """
        向链表尾部添加节点
            尾部节点不存在 新添加的节点为头部和尾部节点
            尾部节点已存在 新添加的节点为新的尾部节点
        :param node: 添加的节点
        :return: 已添加的节点
        """
        # 尾部节点为空
        if not self.tail:
            self.tail = node
            self.head = node
            self.head.next = None
            self.tail.prev = None
        # 尾部节点不为空
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
            self.tail.next = None
        self.size += 1

        return node

    def __remove_head(self):
        """
        删除头部节点
            头部节点不存在 返回None
            头部节点已存在 判断链表节点数量 删除头部节点
        :return: 头部节点
        """
        # 头部节点不存在
        if not self.head:
            return None

        # 链表至少存在两个节点
        head = self.head
        if head.next:
            head.next.prev = None
            self.head = head.next
        # 只存在头部节点
        else:
            self.head = self.tail = None
        self.size -= 1

        return head

    def __remove_tail(self):
        """
        删除尾部节点
            尾部节点不存在 返回None
            尾部节点已存在 判断链表节点数量 删除尾部节点
        :return: 尾部节点
        """
        # 尾部节点不存在
        if not self.tail:
            return None

        # 链表至少存在两个节点
        tail = self.tail
        if tail.prev:
            tail.prev.next = None
            self.tail = tail.prev
        # 只存在尾部节点
        else:
            self.head = self.tail = None
        self.size -= 1

        return tail

    def __remove(self, node):
        """
        删除任意节点
            被删除的节点不存在 默认删除尾部节点
            删除头部节点
            删除尾部节点
            删除其他节点
        :param node: 被删除的节点
        :return: 被删除的节点
        """
        # 被删除的节点不存在
        if not node:
            node = self.tail

        # 删除的是头部节点
        if node == self.head:
            self.__remove_head()
        # 删除的是尾部节点
        elif node == self.tail:
            self.__remove_tail()
        # 删除的既不是头部也不是尾部节点
        else:
            node.next.prev = node.prev
            node.prev.next = node.next
            self.size -= 1

        return node

    def pop(self):
        """
        弹出头部节点
        :return: 头部节点
        """
        return self.__remove_head()

    def append(self, node):
        """
        添加尾部节点
        :param node: 待追加的节点
        :return: 尾部节点
        """
        return self.__add_tail(node)

    def append_front(self, node):
        """
        添加头部节点
        :param node: 待添加的节点
        :return: 已添加的节点
        """
        return self.__add_head(node)

    def remove(self, node=None):
        """
        删除任意节点
        :param node: 待删除的节点
        :return: 已删除的节点
        """
        return self.__remove(node)

    def prints(self):
        """
        打印当前链表
        :return:
        """
        node = self.head
        line = ''
        while node:
            line += '%s' % node
            node = node.next
            if node:
                line += '=>'
        print(line)


class Node(object):

    def __init__(self, key, value):
        """
        初始化方法
        :param key:
        :param value:
        """
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

    def __str__(self):
        val = '{%s: %s}' % (self.key, self.value)
        return val

    def __repr__(self):
        val = '{%s: %s}' % (self.key, self.value)
        return val


class LFUNode(Node):

    def __init__(self, key, value):
        """
        LFU节点 增加频率属性
        :param key:
        :param value:
        """
        self.freq = 0
        super(LFUNode, self).__init__(key, value)


class LFUCache(object):

    def __init__(self, capacity=0xffffffff):
        """
        LFU缓存置换算法 最不经常使用
        :param capacity:
        """
        self.capacity = capacity
        self.size = 0
        self.map = {}
        self.freq_map = {}


    def __update_freq(self, node):
        """
        更新节点频率
        :param node:
        :return:
        """
        freq = node.freq

        # 当前节点所在频率存在 在当前频率链表中移除当前节点
        if freq in self.freq_map:
            node = self.freq_map[freq].remove(node)
            # 当前频率链表为空时删除该频率链表
            if self.freq_map[freq].size == 0:
                del self.freq_map[freq]

        # 将节点按照新频率写入频率链表
        freq += 1
        node.freq = freq
        if freq not in self.freq_map:
            self.freq_map[freq] = DoubleLinkedList()
        self.freq_map[freq].append(node)

        return node

    def get(self, key):
        """
        获取元素
        :return:
        """
        # 节点不存在
        if key not in self.map:
            return None

        # 节点存在 更新使用频率
        old_node = self.map.get(key)
        new_node = self.__update_freq(old_node)
        self.map[key] = new_node

        return new_node.value

    def contains(self, key):
        if key not in self.map:
            return False
        else:
            return True

    def put(self, key, value):
        """
        设置元素
        :param key:
        :param value:
        :return:
        """
        # 节点已存在 更新频率
        if key in self.map:
            old_node = self.map.get(key)
            old_node.value = value
            new_node = self.__update_freq(old_node)
            self.map[key] = new_node
        else:
            # 节点容量达到上限 移除最小频率链表头部的节点
            while self.size + value > self.capacity:
                min_freq = min(self.freq_map)
                node = self.freq_map[min_freq].pop()
                # print(node.value, "被删除的")
                # print('缓存已满，替换')
                # print(self.contains(node.key))
                del self.map[node.key]

                if self.freq_map[min_freq].size == 0:
                    del self.freq_map[min_freq]
                self.size -= node.value

            # 构建新的节点 更新频率
            new_node = LFUNode(key, value)
            new_node = self.__update_freq(new_node)
            self.map[key] = new_node
            self.size += new_node.value

        return new_node

    def prints(self):
        """
        打印当前链表
        :return:
        """
        for freq, link in self.freq_map.items():
            print("frequencies： %d" % freq)
            link.prints()

    def remove(self, key):
        node = self.map.get(key)
        freq = node.freq
        node = self.freq_map[freq].remove(node)

        if freq in self.freq_map:
            del self.map[node.key]
            # 当前频率链表为空时删除该频率链表
            if self.freq_map[freq].size == 0:
                del self.freq_map[freq]
            self.size -= node.value
        return freq


class Cache():
    def __init__(self, size=5):
        self.lfu_cache = LFUCache(5)
        self.remain = size

    def visit(self, chunk_size, key):
        if self.remain - chunk_size >= 0:
            self.lfu_cache.put(key, chunk_size)
            self.remain -= chunk_size

    def contains(self, key):
        return self.lfu_cache.contains(key)


if __name__ == '__main__':

    lfu_cache = LFUCache(30)
    # video_index = np.random.randint(0, 100)
    # video = data_load('./videos/hls_' + video_index.__str__())
    # video_start = np.random.randint(0, len(video[0]) / 2)
    # video_end = np.random.randint(len(video[0]) / 2 + 1, len(video[0]))
    #
    # video_bitrate = np.random.randint(0, 6)
    lfu_cache.put((44, 1, 1), 1)
    lfu_cache.put((44, 2, 1), 1)
    lfu_cache.remove((44, 1, 1))
    lfu_cache.remove((44, 2, 1))
    lfu_cache.put((44, 3, 2), 1)
    lfu_cache.put((44, 4, 3), 1)
    lfu_cache.put((44, 5, 0), 1)
    lfu_cache.put((44, 6, 4), 1)
    print(lfu_cache.contains((44, 7, 3)), 'fdsafdsafdsafd')



    lfu_cache.put(1, 10)
    lfu_cache.put(1, 10)
    lfu_cache.put(2, 10)
    print(lfu_cache.contains(1))
    lfu_cache.put(2, 10)
    print(lfu_cache.contains(1))
    lfu_cache.put(3, 20)
    print(lfu_cache.contains(1))
    lfu_cache.put(3, 10)
    print(lfu_cache.contains(2))
    print(lfu_cache.contains(3))



# coding: UTF-8
# 数据来源：http://www.di.unipi.it/optimize/Data/MEX.html
# sscflp(Capacitated Facility Location Problem)

import numpy as np
import re
import time
import csv
import os


class Solution:

    # tag是标记
    # filename是计算的数据来源
    # mode可以是
    #   "greed"(贪心)
    #   "la"(局部搜索——爬山法)
    #   "sa"(模拟退火法)
    # csv_writer是将结果写入文件的csv.writer，用于记录每个样例的
    #   最小消费
    #   计算用时
    # result_file是用于写入具体结果的文件，会记录每个样例的三行信息：
    #   最小消费
    #   工厂启动情况
    #   用户对工厂的选择情况
    def __init__(self, tag, filename, mode="greed", csv_writer=None, result_file=None):
        file = open(filename, "r")
        lines = file.read().splitlines()
        # 首先读取工厂数量以及用户数量
        line = re.findall("\d+", lines.pop(0))
        factory_num, user_num = int(line[0]), int(line[1])
        # factories里包含每个工厂的状态，每个item有两个元素，第一个是工厂容量，第二个是启动工厂的消费
        self.factories = np.zeros([factory_num, 2], dtype=np.int)
        for i in range(factory_num):
            line = re.findall("\d+", lines.pop(0))
            self.factories[i] = [int(line[0]), int(line[1])]
        # 每个用户需要消耗的工厂容量
        user_volume = []
        while len(user_volume) < user_num:
            user_volume.extend(np.array(re.findall("\d+", lines.pop(0))).astype(np.int).tolist())
        # 每个用户使用对应工厂的消费，每个用户使用不同工厂的费用不用
        self.users = np.zeros([user_num, factory_num + 1], dtype=np.int)
        for i in range(user_num):
            user = [user_volume[i]]
            while len(user) < factory_num + 1:
                user.extend(np.array(re.findall("\d+", lines.pop(0))).astype(np.int).tolist())
            self.users[i] = user
        file.close()
        # 计算的总消费结果、工厂启动的情况、用户选择使用的工厂、是否选择正确的算法
        result, factory_state, user_factory, flag = None, None, None, True
        start = time.time()
        if mode == "greed":
            result, factory_state, user_factory = self.greed_answer(tag)
        elif mode == "la":
            result, factory_state, user_factory = self.la_answer(tag)
        elif mode == "sa":
            result, factory_state, user_factory = self.sa_answer(tag)
        else:
            print("something wrong happen!")
            flag = False
        # 计算耗时
        time_cost = time.time() - start
        print("用时", time_cost, "\n")
        # 将结果写入文件
        if csv_writer is not None and flag:
            # 记录该样例的最小消费以及计算耗时
            csv_writer.writerow([str(tag), str(result), str(time_cost)])
        # 将具体结果写入文件
        if result_file is not None and flag:
            # 记录该样例的最小消费、工厂启用情况、用户选择情况
            result_file.write("p{0}".format(tag) + "\n" + str(result) + "\n"
                              + ' '.join(np.array(factory_state).astype(np.str).tolist())
                              + "\n" + ' '.join(np.array(user_factory).astype(np.str).tolist()) + "\n\n")

    # 贪心算法，tag是标记
    def greed_answer(self, tag):
        # 为了不让启动工厂的消费影响到用户的选择情况，在所有工厂开启的情况下进行贪心。
        # 当然，这只是一种效果似乎不错的贪心策略，应该还有其他的更好的贪心策略选择。
        factory_states = np.ones(self.factories.shape[0], np.int).tolist()
        best_result, user_factory, num = self.calculate(factory_states)
        print("greed--result{0}".format(tag), best_result, num)
        return best_result, factory_states, user_factory

    # 多领域操作爬山法——基于工厂与贪心，tag是标记
    def la_answer(self, tag):
        # 开始时工厂启用情况
        factory_num = self.factories.shape[0]
        factory_states = np.ones([factory_num], np.int).tolist()
        # 决定爬山法爬山次数的外层循环的参数
        current_temperature = 2 * factory_num // 10
        final_temperature = 0.5 / (factory_num // 10)
        # 初始的 消费、用户选择情况、无法选上工厂的用户的数量
        best_result, best_user_factory, best_num = self.calculate(factory_states)
        correct_flag = True
        while current_temperature > final_temperature:
            for _ in range(factory_num):
                num1, num2 = np.random.randint(low=0, high=10, size=2).tolist()
                while num1 == num2:
                    num1, num2 = np.random.randint(low=0, high=10, size=2).tolist()
                if num1 > num2:
                    num1, num2 = num2, num1
                # 让启用的工厂总能满足用户的需求
                if not correct_flag:
                    zero_indexes = np.where(np.array(factory_states) == 0)[0]
                    length = zero_indexes.shape[0]
                    if length > 0:
                        factory_states[zero_indexes[np.random.randint(low=0, high=length, size=1)[0]]] = 1
                        correct_flag = True

                # 用1减一个数，即如果一个工厂启动了，那么关闭它，否则启动它
                temp_states = factory_states.copy()
                temp_states[num1] = 1 - temp_states[num1]
                temp_result, temp_user_factory, temp_num = self.calculate(temp_states)
                if best_result > temp_result > 0 and temp_num == 0:
                    best_result, best_user_factory, best_num = temp_result, temp_user_factory, temp_num
                    factory_states = temp_states
                elif temp_result == -1:
                    correct_flag = False

                # 交换两个数，即交换两个工厂的启动情况
                temp_states = factory_states.copy()
                temp_states[num1], temp_states[num2] = temp_states[num2], temp_states[num1]
                temp_result, temp_user_factory, temp_num = self.calculate(temp_states)
                if best_result > temp_result > 0 and temp_num == 0:
                    best_result, best_user_factory, best_num = temp_result, temp_user_factory, temp_num
                    factory_states = temp_states
                elif temp_result == -1:
                    correct_flag = False

                # 将num2位置的数插到num1位置，改变num1到num2间所有工厂的启动情况
                temp_states = factory_states.copy()
                _temp = temp_states[num1]
                for i in range(num1, num2):
                    temp_states[i] = temp_states[i + 1]
                temp_states[num2] = _temp
                temp_result, temp_user_factory, temp_num = self.calculate(temp_states)
                if best_result > temp_result > 0 and temp_num == 0:
                    best_result, best_user_factory, best_num = temp_result, temp_user_factory, temp_num
                    factory_states = temp_states
                elif temp_result == -1:
                    correct_flag = False

                # 将num2与num1间的数逆转 -- 懒得实现了！！！
            # 参数变化
            current_temperature = current_temperature * 0.98
        print("la--result{0}".format(tag), best_result, best_num)
        return best_result, factory_states, best_user_factory

    # 模拟退火法——基于工厂启动状态与贪心，tag是标记
    def sa_answer(self, tag):
        # 开始时工厂启动情况
        factory_num = self.factories.shape[0]
        factory_states = np.ones([factory_num], np.int).tolist()
        # 初温与末温
        current_temperature = 2 * factory_num // 10
        final_temperature = 0.5 / (factory_num // 10)
        # 初始的 消费、用户选择情况、无法选上工厂的用户的数量
        now_result, now_user_factory, now_num = self.calculate(factory_states)
        # 维护一个全局的最优解
        best_result, best_user_factory, best_num = now_result, now_user_factory.copy(), now_num
        best_factory_states = factory_states.copy()
        # 这个flag是让工厂总能满足用户的需求
        correct_flag = True
        while current_temperature > final_temperature:
            for _ in range(factory_num):
                num1, num2 = np.random.randint(low=0, high=10, size=2).tolist()
                while num1 == num2:
                    num2 = np.random.randint(low=0, high=10)
                if num1 > num2:
                    num1, num2 = num2, num1
                if not correct_flag:
                    zero_indexes = np.where(np.array(factory_states) == 0)[0]
                    length = zero_indexes.shape[0]
                    if length > 0:
                        factory_states[zero_indexes[np.random.randint(low=0, high=length, size=1)[0]]] = 1
                        correct_flag = True

                temp_states, temp_result, temp_user_factory, temp_num = None, 10000000, None, None

                # 用1减一个数，即如果一个工厂启动了，那么关闭它，否则启动它
                temp_states1 = factory_states.copy()
                temp_states1[num1] = 1 - temp_states1[num1]
                temp_result1, temp_user_factory1, temp_num1 = self.calculate(temp_states1)
                if temp_num1 == 0 and temp_result > temp_result1 > 0:
                    temp_states = temp_states1
                    temp_result, temp_user_factory, temp_num = temp_result1, temp_user_factory1, temp_num1

                # 交换两个数，即交换两个工厂的启动情况
                temp_states2 = factory_states.copy()
                temp_states2[num1], temp_states2[num2] = temp_states2[num2], temp_states2[num1]
                temp_result2, temp_user_factory2, temp_num2 = self.calculate(temp_states2)
                if temp_num2 == 0 and temp_result > temp_result2 > 0:
                    temp_states = temp_states2
                    temp_result, temp_user_factory, temp_num = temp_result2, temp_user_factory2, temp_num2

                # 将num2位置的数插到num1位置，改变num1到num2间所有工厂的启动情况
                temp_states3 = factory_states.copy()
                _temp = temp_states3[num1]
                for i in range(num1, num2):
                    temp_states3[i] = temp_states3[i + 1]
                temp_states3[num2] = _temp
                temp_result3, temp_user_factory3, temp_num3 = self.calculate(temp_states3)
                if temp_num3 == 0 and temp_result > temp_result3 > 0:
                    temp_states = temp_states3
                    temp_result, temp_user_factory, temp_num = temp_result3, temp_user_factory3, temp_num3

                diff = best_result - temp_result
                if temp_num == 0 and temp_result > 0 and (diff > 0 or np.exp(
                        -1 * np.abs(diff) / current_temperature) >= np.random.rand()):
                    now_result, now_user_factory, now_num = temp_result, temp_user_factory, temp_num
                    factory_states = temp_states
                    if now_result < best_result:
                        best_result, best_user_factory, best_num = now_result, now_user_factory.copy(), now_num
                        best_factory_states = factory_states.copy()
                elif temp_result == -1:
                    correct_flag = False

                # 将num2与num1间的数逆转--懒得实现了！！！
            # 温度变化
            current_temperature = current_temperature * 0.98
        print("sa--result{0}".format(tag), best_result, best_num)
        return best_result, best_factory_states, best_user_factory

    # 根据工厂状态计算消费——基于贪心
    def calculate(self, factory_states):
        # 先获取启动了的工厂，根据启动了的工厂限制用户选择
        factories = []
        correct_indexes = []
        users = [self.users[:, 0]]
        for i in range(len(factory_states)):
            if factory_states[i] == 1:
                factories.append(self.factories[i])
                correct_indexes.append(i)
                users.append(self.users[:, i + 1])
        # 如果没有启动的工厂，直接返回
        if len(factories) == 0:
            return 10000000, None, None
        factories = np.array(factories)
        users = np.vstack(users).T
        result = factories[:, 1].sum()
        volumes = factories[:, 0].copy()
        # 如果启动了的工厂容量不足，直接返回
        if volumes.sum() < users[:, 0].sum():
            return -1, None, None
        # 依据贪心计算消费
        user_factory = []
        num = 0
        for _user in users:
            user, max = _user.copy(), _user.max()
            user_volume, user_cost = user[0], user[1:]
            flag = True
            while flag:
                min, index = user_cost.min(), np.where(user_cost==np.min(user_cost))[0][0]
                if max == min:
                    num += 1
                    flag = False
                if volumes[index] >= user_volume:
                    volumes[index] -= user_volume
                    result += min
                    user_factory.append(correct_indexes[index])
                    break
                else:
                    user_cost[index] = max
        # 返回消费、用户对工厂的选择、没法选择到工厂的用户数量
        return result, user_factory, num

    # 模拟退火法——基于用户选择情况
    def sa_answer2(self, tag):
        pass

    # 线性规划
    def lp_answer(self, tag):
        pass

    # 遗传算法
    def ga_answer(self, tag):
        pass

    # 禁忌搜索
    def tabu_answer(self, tag):
        pass


if __name__ == "__main__":
    # 创建结果文件夹
    result_path = os.getcwd() + "\\result"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # 实例化对应的文件管理对象，用于写入结果
    greed_csv_file = open("result/greed_table_result.csv", mode="w+", newline='', encoding="utf-8")
    la_csv_file = open("result/la_table_result.csv", mode="w+", newline='', encoding="utf-8")
    sa_csv_file = open("result/sa_table_result.csv", mode="w+", newline='', encoding="utf-8")
    greed_result_file = open("result/greed_detail_result.txt", mode="w+", encoding="utf-8")
    la_result_file = open("result/la_detail_result.txt", mode="w+", encoding="utf-8")
    sa_result_file = open("result/sa_detail_result.txt", mode="w+", encoding="utf-8")
    greed_writer = csv.writer(greed_csv_file, dialect='excel', delimiter=',')
    la_writer = csv.writer(la_csv_file, dialect='excel', delimiter=',')
    sa_writer = csv.writer(sa_csv_file, dialect='excel', delimiter=',')
    fields = ["index", "result", "time(s)"]
    greed_writer.writerow(fields)
    la_writer.writerow(fields)
    sa_writer.writerow(fields)
    # 将各个算法的结果写入文件
    base_instances = "./Instances/p{0}"
    for i in range(1, 72):
        # 贪心
        Solution(i, base_instances.format(i), mode="greed", csv_writer=greed_writer, result_file=greed_result_file)
        # 局部搜索——爬山法
        Solution(i, base_instances.format(i), mode="la", csv_writer=la_writer, result_file=la_result_file)
        # 模拟退火法
        Solution(i, base_instances.format(i), mode="sa", csv_writer=sa_writer, result_file=sa_result_file)
    # 关闭各个文件
    greed_csv_file.close()
    la_csv_file.close()
    sa_csv_file.close()
    greed_result_file.close()
    la_result_file.close()
    sa_result_file.close()

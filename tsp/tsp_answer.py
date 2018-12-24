# coding: UTF-8

import copy
import time
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import threading
from multiprocessing import Process
from matplotlib import animation

# http://www.cnblogs.com/zealousness/p/9753282.html
# https://blog.csdn.net/u012513972/article/details/78389863
# https://www.jianshu.com/p/ae5157c26af9
# https://blog.csdn.net/qq_38788128/article/details/80804661
# https://blog.csdn.net/ty_0930/article/details/52119075
# https://blog.csdn.net/icurious/article/details/81709797
# https://www.cnblogs.com/apexchu/p/5015961.html
# https://www.cnblogs.com/zhoujie/p/nodejs2.html
# https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/
# https://blog.csdn.net/u012750702/article/details/54563515


class Solution:

    def __init__(self, filename="d198.tsp", modify_flag=False, ga_flag=False, sa_flag=False, sa_flag2=False, num=1):
        file = open(filename, "r")
        lines = file.read().splitlines()
        file.close()
        # 获取城市数量，并准备存储数据的结构，城市编号默认从0~city_num - 1
        city_num = np.int(lines.pop(0))
        self.cities = np.zeros([city_num, 2], dtype=np.float)
        self.city_states = np.zeros([city_num, city_num], dtype=np.float)
        # 城市分布坐标
        for i in range(city_num):
            self.cities[i] = np.array(lines.pop(0).split(" ")).astype(np.float)
        # 城市邻接矩阵
        for i in range(city_num):
            x1, y1 = self.cities[i, 0], self.cities[i, 1]
            for j in range(city_num):
                x2, y2 = self.cities[j, 0], self.cities[j, 1]
                self.city_states[i, j] = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))
        # 各个算法的所有案例的结果
        results = {"ga_result": list(), "modify_result": list(), "sa_result": list(), "sa_result2": list()}
        for i in range(num):
            start = time.time()
            # 改良圈算法
            if modify_flag:
                results["modify_result"].append(self.modify_circle(i))
            # 遗传算法
            if ga_flag:
                results["ga_result"].append(self.ga_answer(50, 50, 0.2, 1, 10000, i))
            # 模拟退火法
            if sa_flag:
                results["sa_result"].append(self.sa_answer(i))
            if sa_flag2:
                results["sa_result2"].append(self.sa_answer2(i))
            print("消耗时间", time.time() - start)
        # print(results)
        if modify_flag:
            _results = results["modify_result"]
            for i in range(len(_results)):
                print("修改圈算法结果{0}".format(i), _results[i])
            print("均值", np.average(_results))
        if ga_flag:
            _results = results["ga_result"]
            for i in range(len(_results)):
                print("遗传算法结果{0}".format(i), _results[i])
            print("均值", np.average(_results))
        if sa_flag:
            _results = results["sa_result"]
            for i in range(len(_results)):
                print("模拟退火算法结果{0}".format(i), _results[i])
            print("均值", np.average(_results))
        if sa_flag2:
            _results = results["sa_result2"]
            for i in range(len(_results)):
                print("模拟退火算法结果{0}".format(i), _results[i])
            print("均值", np.average(_results))

    # 改良圈算法
    def modify_circle(self, tag):
        city_num = self.city_states.shape[0]
        initial = list(range(city_num))
        initial = np.array(initial + [0])
        all_result = [initial]
        # print(np.sum([self.city_states[initial[i], initial[i + 1]] for i in range(city_num)]))
        for k in range(city_num):
            flag = 0    # 退出标志
            for m in range(city_num - 2):
                initial_m, initial_m2 = initial[m], initial[m + 1]
                for n in range(m + 2, city_num):
                    initial_n, initial_n2 = initial[n], initial[n + 1]
                    if self.city_states[initial_m, initial_n] + self.city_states[initial_m2, initial_n2] < \
                            self.city_states[initial_m, initial_m2] + self.city_states[initial_n, initial_n2]:
                        initial[m + 1:n + 1] = initial[n:m:-1]
                        all_result.append(initial.copy())
                        flag += 1
            if flag == 0:
                break
        cost = np.sum([self.city_states[initial[i], initial[i + 1]] for i in range(city_num)])
        # show_thread = threading.Thread(target=self.show, args=(all_initials,))
        print("results长度", len(all_result))
        show_thread = Process(target=self.show, args=(all_result, tag, ))
        show_thread.start()
        # self.show(all_result)
        return cost

    # 遗传算法: 群体规模、子代规模、变异概率、杂交概率、遗传次数
    def ga_answer(self, total_num, son_num, variation, cross, ga_num, tag):
        # 改良圈算法得到初始解
        totals = []
        totals_cost = []
        city_num = self.city_states.shape[0]
        num1 = total_num    # 改良圈的，压不下去？？？
        num2 = 0    # 0-197的，顺序着来的
        num3 = total_num - num1 - num2  # 贪心
        for i in range(num1):
            temp = list(range(1, city_num))
            np.random.shuffle(temp)
            temp = np.array([0] + temp + [0])
            # for k in range(city_num):
            #     flag = 0  # 退出标志
            #     for m in range(city_num - 2):
            #         initial_m, initial_m2 = temp[m], temp[m + 1]
            #         for n in range(m + 2, city_num):
            #             initial_n, initial_n2 = temp[n], temp[n + 1]
            #             if self.city_states[initial_m, initial_n] + self.city_states[initial_m2, initial_n2] < \
            #                     self.city_states[initial_m, initial_m2] + self.city_states[initial_n, initial_n2]:
            #                 temp[m + 1:n + 1] = temp[n:m:-1]
            #                 flag += 1
            #     if flag == 0:
            #         break
            totals.append(temp)
            cost = np.sum([self.city_states[temp[i], temp[i + 1]] for i in range(city_num)])
            totals_cost.append(cost)
        # 0-197的初始解
        templet = np.array([0] + list(range(1, city_num)) + [0])
        templet_cost = np.sum([self.city_states[templet[i], templet[i + 1]] for i in range(city_num)])
        for i in range(num2):
            totals.append(copy.deepcopy(templet))
            totals_cost.append(templet_cost)
        # 贪心得到初始解
        greed_path = [0]
        rest = list(range(1, city_num))
        while len(greed_path) < city_num:
            node1 = greed_path[len(greed_path) - 1]
            min_cost = self.city_states[node1][rest[0]]
            min_index = 0
            for i in range(1, len(rest)):
                now_cost = self.city_states[node1][rest[i]]
                if min_cost > now_cost:
                    min_cost = now_cost
                    min_index = i
            greed_path.append(rest.pop(min_index))
        greed_path = np.array(greed_path + [0])
        greed_cost = np.sum([self.city_states[greed_path[i], greed_path[i + 1]] for i in range(city_num)])
        for i in range(num3):
            totals.append(copy.deepcopy(greed_path))
            totals_cost.append(greed_cost)
        # 遗传算法
        best_result = np.min(totals_cost)
        all_result = [totals[np.where(totals_cost==best_result)[0][0]]]
        # print(best_result)
        for _ga_num in range(ga_num):
            best_index = np.where(totals_cost==best_result)[0]
            if best_index.shape[0] > 0.3:
                all_result.append(totals[best_index[0]])
            sons = []
            sons_cost = []
            # 轮盘赌选择法
            probabilities = np.sum(totals_cost) / totals_cost
            probabilities = probabilities / np.sum(probabilities)
            for _ in range(0, son_num, 2):
                [father_index, mother_index] = np.random.choice(np.arange(len(totals)), size=2, replace=False, p=probabilities)
                father, mother = totals[father_index], totals[mother_index]
                # 交叉
                if cross <= np.random.rand():
                    continue
                probability = np.random.rand()
                if probability < 0.5:
                    # 解决冲突
                    [start, end] = np.random.choice(list(range(1, city_num)), size=2, replace=False)
                    while start >= end:
                        [start, end] = np.random.choice(list(range(1, city_num)), size=2, replace=False)
                    _conflicts = {father[i]: mother[i] for i in range(start, end)}
                    conflict_keys = _conflicts.keys()
                    for key in conflict_keys:
                        temp = _conflicts[key]
                        while temp in conflict_keys:
                            _conflicts[key] = _conflicts[temp]
                            _conflicts[temp] = 0
                            temp = _conflicts[key]
                    conflicts = dict()
                    for key, value in _conflicts.items():
                        if value > 0:
                            conflicts[key] = value
                    # 真正交配
                    son1, son2 = father.copy(), mother.copy()
                    son1[start:end], son2[start:end] = son2[start:end].copy(), son1[start:end].copy()
                    # _son1, _son2 = son1.copy(), son2.copy()
                    # 解决冲突2
                    for key, value in conflicts.items():
                        for index in np.where(son1 == value)[0]:
                            if index >= end or index < start:
                                son1[index] = key
                        for index in np.where(son2 == key)[0]:
                            if index >= end or index < start:
                                son2[index] = value
                else:
                    index = np.random.randint(low=1, high=city_num)
                    son1, son2 = father.copy(), mother.copy()
                    son1[np.where(son1 == son2[index])[0][0]] = son1[index]
                    son2[np.where(son2 == son1[index])[0][0]] = son2[index]
                    son1[index], son2[index] = son2[index], son1[index]
                sons.extend([son1, son2])
                sons_cost.append(np.sum([self.city_states[son1[i], son1[i + 1]] for i in range(city_num)]))
                sons_cost.append(np.sum([self.city_states[son2[i], son2[i + 1]] for i in range(city_num)]))
            best_result = np.min([best_result] + sons_cost)
            # 变异: 逆转
            sons_cost.extend(totals_cost)
            sons.extend(totals)
            son_range = range(len(sons))
            for i in son_range:
                if np.random.random() <= variation:
                    son = sons[i].copy()
                    [index1, index2] = np.random.choice(list(range(1, city_num)), size=2, replace=False)
                    if index1 > index2:
                        index1, index2 = index2, index1
                    # son[index1], son[index2] = son[index2], son[index1]
                    # son[index1: index2] = son[index2 - 1: index1 - 1: -1]
                    probability = np.random.rand() * 5
                    if 0 <= probability < 1:
                        son[index1], son[index2] = son[index2], son[index1]
                    # 将index2的城市插入到index1前
                    elif 1 <= probability < 2:
                        temp = son[index2]
                        for j in range(index2 - index1):
                            son[index2 - j] = sons[i][index2 - j - 1]
                        son[index1] = temp
                    # 将index1与index2间的城市逆转
                    else:
                        for j in range(index2 - index1 + 1):
                            son[index1 + j] = sons[i][index2 - j]
                    son_cost = np.sum([self.city_states[son[i], son[i + 1]] for i in range(city_num)])
                    if sons_cost[i] > son_cost or _ga_num > ga_num * 0.5:
                        sons_cost[i] = son_cost
                        sons[i] = son
            best_result = np.min([best_result] + sons_cost)
            # 适者生存
            # temp_results = {sons_cost[i]: sons[i] for i in son_range}
            # temp_results = sorted(temp_results.items(), key=lambda t: t[0])
            temp_results = [(sons_cost[i], sons[i]) for i in son_range]
            temp_results = sorted(temp_results, key=lambda t: t[0])
            totals = [temp_results[i][1] for i in range(total_num)]
            totals_cost = [temp_results[i][0] for i in range(total_num)]
            print("遗传次数", _ga_num, "当前局部最优解", min(totals_cost), "当前全局最优解", best_result)
        # 选出最优者
        best_result = np.min([best_result] + totals_cost)
        print(best_result)
        print("results长度", len(all_result))
        show_thread = Process(target=self.show, args=(all_result, tag,))
        show_thread.start()
        return best_result

    # 模拟退火法 -- 廖志勇
    def sa_answer2(self, tag):
        # 先进行贪心取得初始解
        city_num = self.city_states.shape[0]
        greed_path = [0]
        rest = list(range(1, city_num))
        while len(greed_path) < city_num:
            node1 = greed_path[len(greed_path) - 1]
            min_cost = self.city_states[node1][rest[0]]
            min_index = 0
            for i in range(1, len(rest)):
                now_cost = self.city_states[node1][rest[i]]
                if min_cost > now_cost:
                    min_cost = now_cost
                    min_index = i
            greed_path.append(rest.pop(min_index))
        greed_path.append(0)
        all_result = [greed_path]
        cost = np.sum([self.city_states[greed_path[i], greed_path[i + 1]] for i in range(city_num)])
        best_cost = cost
        current_temperature = 1
        while current_temperature > 0.00001:
            nochange = 0
            for _ in range(300):
                index1, index2 = np.random.randint(1, city_num, size=2).tolist()
                while index1 == index2:
                    index2 = np.random.randint(1, city_num)
                if index2 < index1:
                    index1, index2 = index2, index1
                path1, path2, path3 = copy.deepcopy(greed_path), copy.deepcopy(greed_path), copy.deepcopy(greed_path)
                probability = np.random.rand() * 5
                # 交换index1与index2位置的城市
                if 0 <= probability < 1:
                    path1[index1], path1[index2] = path1[index2], path1[index1]
                    cost1 = np.sum([self.city_states[path1[i], path1[i + 1]] for i in range(city_num)])
                    result_path = path1
                    result_cost = cost1
                # 将index2的城市插入到index1前
                elif 1 <= probability < 2:
                    # path2.insert(index2, path2.pop(index1))
                    temp = path2[index2]
                    for i in range(index2 - index1):
                        path2[index2 - i] = greed_path[index2 - i - 1]
                    path2[index1] = temp
                    cost2 = np.sum([self.city_states[path2[i], path2[i + 1]] for i in range(city_num)])
                    result_path = path2
                    result_cost = cost2
                # 将index1与index2间的城市逆转
                else:
                    # path3[index1: index2] = path3[index2 - 1: index1 - 1: -1]
                    for i in range(index2 - index1 + 1):
                        path3[index1 + i] = greed_path[index2 - i]
                    cost3 = np.sum([self.city_states[path3[i], path3[i + 1]] for i in range(city_num)])
                    result_path = path3
                    result_cost = cost3
                if result_cost < cost or np.exp(-1 * np.abs(cost - result_cost) / current_temperature) >= np.random.rand():
                    greed_path = result_path
                    cost = result_cost
                    if cost < best_cost:
                        best_path = copy.deepcopy(greed_path)
                        best_cost = cost
                        all_result.append(best_path)
                    nochange = 0
                else:
                    nochange += 1
                if nochange >= 90:
                    break
            # print("当前局部最优", cost, "当前最优解", best_cost, "当前温度", current_temperature)
            current_temperature *= 0.99
        print(best_cost)
        # print("results长度", len(all_result))
        show_thread = Process(target=self.show, args=(all_result, tag, ))
        show_thread.start()
        return best_cost

    # 模拟退火法 -- 梁毓颖
    def sa_answer(self, tag):
        # 先进行贪心
        city_num = self.city_states.shape[0]
        greed_path = [0]
        rest = list(range(1, city_num))
        greed_path.extend(rest)
        greed_path.append(0)
        all_result = [greed_path]
        cost = np.sum([self.city_states[greed_path[i], greed_path[i + 1]] for i in range(city_num)])
        best_cost = cost
        # print(cost)
        current_temperature = 100
        while current_temperature > 0.1:
            for _ in range(1000):
                index1, index2 = np.random.randint(1, city_num, size=2).tolist()
                while index1 == index2:
                    index2 = np.random.randint(1, city_num)
                if index2 < index1:
                    index1, index2 = index2, index1
                path1, path2, path3 = copy.deepcopy(greed_path), copy.deepcopy(greed_path), copy.deepcopy(greed_path)
                # 交换index1与index2位置的城市
                path1[index1], path1[index2] = path1[index2], path1[index1]
                cost1 = np.sum([self.city_states[path1[i], path1[i + 1]] for i in range(city_num)])
                result_path = path1
                result_cost = cost1
                # 将index2的城市插入到index1前
                # path2.insert(index1, path2.pop(index2))
                temp = path2[index2]
                for i in range(index2 - index1):
                    path2[index2 - i] = greed_path[index2 - i - 1]
                path2[index1] = temp
                cost2 = np.sum([self.city_states[path2[i], path2[i + 1]] for i in range(city_num)])
                if result_cost > cost2:
                    result_path = path2
                    result_cost = cost2
                # 将index1与index2间的城市逆转
                # path3[index1: index2] = path3[index2 - 1: index1 - 1: -1]
                for i in range(index2 - index1 + 1):
                    path3[index1 + i] = greed_path[index2 - i]
                cost3 = np.sum([self.city_states[path3[i], path3[i + 1]] for i in range(city_num)])
                if result_cost > cost3:
                    result_path = path3
                    result_cost = cost3
                # 选出最优解
                if result_cost < cost or np.exp(-1 * np.abs(cost - result_cost) / current_temperature) >= np.random.rand():
                    greed_path = result_path
                    cost = result_cost
                    if cost < best_cost:
                        best_path = copy.deepcopy(greed_path)
                        best_cost = cost
                        all_result.append(best_path)
            current_temperature *= 0.95
            # print("当前局部最优", cost, "当前最优解", best_cost, "当前温度", current_temperature)
        print("模拟退火法结果{0}".format(tag), best_cost)
        # print("results长度", len(all_result))
        show_thread = Process(target=self.show, args=(all_result, tag, ))
        show_thread.start()
        return best_cost

    # 画动图/保存视频
    def show(self, all_result, tag):
        # time.sleep(10)
        len1 = len(all_result)
        len2 = min(300, len1)
        sampling = np.floor(np.linspace(0, len1 - 1, len2, endpoint=True)).astype(np.int)
        figure, ax = plt.subplots()
        ax.scatter(self.cities[:, 0], self.cities[:, 1])
        _line = np.array([self.cities[i] for i in all_result[0]])
        line, = ax.plot(_line[:, 0], _line[:, 1], color="r")

        def init():
            return line

        def update(frame):
            frame = all_result[frame]
            _line2 = np.array([self.cities[i] for i in frame])
            line.set_ydata(_line2[:, 1])
            line.set_xdata(_line2[:, 0])
            return line

        anim = animation.FuncAnimation(fig=figure, func=update, init_func=init, interval=50, frames=sampling,
                                       repeat=False)
        # Set up formatting for the movie files
        # writer = animation.writers['ffmpeg'](fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # anim.save("video{0}.mp4".format(tag), writer=writer)

        # plt.ion()
        # plt.pause(10)
        # plt.close("all")

        plt.title("figure{0}".format(tag))
        plt.show()

    # 计算一条路径
    def calculate(self, _list):
        result = np.sum([self.city_states[_list[i], _list[i + 1]] for i in range(self.city_states.shape[0])])
        print(result)
        self.show([_list], 0)
        return result


if __name__ == "__main__":
    continue_flag = True
    while continue_flag:
        print("这里有四个选择可以用于解TSP问题，你可以选择其中一个用于解答TSP问题，你也可以输入quit离开这里：\n"
              "    A. 修改圈算法\n"
              "    B. 遗传算法\n"
              "    C. 模拟退火法1\n"
              "    D. 模拟退火法2\n"
              "请选择其中一个编号，并按照\"编号 样例个数\"的格式输入：\n")
        command = input()
        if command == "quit":
            break
        args = command.split()
        print()
        if args[0] == "A":
            # Solution(args[1], modify_flag=True, num=int(args[2]))
            Solution(modify_flag=True, num=int(args[1]))
        if args[0] == "B":
            # Solution(args[1], ga_flag=True, num=int(args[2]))
            Solution(ga_flag=True, num=int(args[1]))
        if args[0] == "C":
            # Solution(args[1], sa_flag=True, num=int(args[2]))
            Solution(sa_flag=True, num=int(args[1]))
        if args[0] == "D":
            # Solution(args[1], sa_flag2=True, num=int(args[2]))
            Solution(sa_flag2=True, num=int(args[1]))
        print()

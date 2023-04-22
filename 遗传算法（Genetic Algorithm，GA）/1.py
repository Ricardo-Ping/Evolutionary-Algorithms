import numpy as np
from sklearn.metrics import mean_squared_error

class MatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=20, lr=0.01, reg=0.01, n_epochs=10, batch_size=100):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def fit(self, train_matrix):
        # 初始化用户和物品向量
        self.user_vectors = np.random.normal(size=(self.n_users, self.n_factors))
        self.item_vectors = np.random.normal(size=(self.n_items, self.n_factors))

        # 创建损失历史记录数组
        self.loss_history = []

        # 按批次迭代训练模型
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for i, (u, i, r) in enumerate(self._iterate_minibatches(train_matrix)):
                e_ui = r - np.dot(self.user_vectors[u], self.item_vectors[i].T)
                self.user_vectors[u] += self.lr * (e_ui[:, np.newaxis] * self.item_vectors[i] - self.reg * self.user_vectors[u])
                self.item_vectors[i] += self.lr * (e_ui[:, np.newaxis] * self.user_vectors[u] - self.reg * self.item_vectors[i])
                epoch_loss += np.sum(e_ui ** 2) + 0.5 * self.reg * (np.sum(self.user_vectors[u] ** 2) + np.sum(self.item_vectors[i] ** 2))
            epoch_loss /= len(train_matrix.data)
            self.loss_history.append(epoch_loss)
            print(f'Epoch {epoch + 1}/{self.n_epochs}: loss = {epoch_loss:.4f}')

    def predict(self, X_test):
        pred_scores = np.dot(self.user_vectors, self.item_vectors.T)
        pred_scores_flat = pred_scores[X_test['user_id'].values - 1, X_test['item_id'].values - 1]
        return pred_scores_flat

    def _iterate_minibatches(self, train_matrix):
        indices = np.arange(len(train_matrix.data))
        np.random.shuffle(indices)
        for start_idx in range(0, len(train_matrix.data), self.batch_size):
            excerpt = indices[start_idx:start_idx + self.batch_size]
            yield train_matrix.row[excerpt], train_matrix.col[excerpt], train_matrix.data[excerpt]

class GeneticAlgorithm:
    def __init__(self, n_users, n_items, pop_size=10, n_generations=50, crossover_rate=0.8, mutation_rate=0.2):
        self.n_users = n_users
        self.n_items = n_items
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def evolve(self, train_matrix, X_test, y_test):
        # 创建初始种群
        pop = self._create_pop()

        # 迭代遗传算法
        for gen in range(self.n_generations):
            # 计算适应度值
            fitness_vals = [self.fitness(chrom, train_matrix, X_test, y_test) for chrom in pop]

            # 选择父代并创建子代
            parents = self._selection(pop, fitness_vals)
            offspring = self._crossover(parents)
            offspring = self._mutate(offspring)

            # 替换当前种群
            pop = parents + offspring

            # 输出每代的最佳适应度值
            best_fitness = np.max(fitness_vals)
            print(f'Generation {gen + 1}/{self.n_generations}: best fitness = {best_fitness:.4f}')

        # 返回最终种群中的最佳个体
        best_idx = np.argmax(fitness_vals)
        return fitness_vals[best_idx], pop[best_idx]

    def fitness(self, chrom, train_matrix, X_test, y_test):
        user_vectors, item_vectors = self.decode(chrom)
        mf = MatrixFactorization(self.n_users, self.n_items, n_factors=user_vectors.shape[1])
        mf.user_vectors = user_vectors
        mf.item_vectors = item_vectors
        pred_scores = mf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_scores))
        return 1.0 / rmse

    def decode(self, chrom):
        # 将染色体解码为用户和物品向量矩阵
        cutoff = int(np.ceil(self.n_factors * self.n_items / (self.n_users + self.n_items)))
        user_chrom = chrom[:cutoff]
        item_chrom = chrom[cutoff:]
        user_vectors = np.reshape(user_chrom, (self.n_users, self.n_factors))
        item_vectors = np.reshape(item_chrom, (self.n_items, self.n_factors))
        return user_vectors, item_vectors

    def _create_pop(self):
        # 创建初始种群
        pop = []
        for i in range(self.pop_size):
            chrom_len = self.n_factors * (self.n_users + self.n_items)
            chrom = np.random.normal(size=chrom_len)
            pop.append(chrom)
        return pop

    def _selection(self, pop, fitness_vals):
        # 选择父代
        idx = np.argsort(fitness_vals)[::-1] # 按适应度值从大到小排序
        parents = [pop[i] for i in idx[:2]] # 选择两个最好的个体作为父代
        return parents

    def _crossover(self, parents):
        # 执行单点交叉
        if np.random.rand() < self.crossover_rate:
            chrom_len = len(parents[0])
            crossover_pt = np.random.randint(1, chrom_len)
            offspring1 = np.concatenate((parents[0][:crossover_pt], parents[1][crossover_pt:]))
            offspring2 = np.concatenate((parents[1][:crossover_pt], parents[0][crossover_pt:]))
            return [offspring1, offspring2]
        else:
            return []

    def _mutate(self, offspring):
        # 执行高斯突变
        for chrom in offspring:
            if np.random.rand() < self.mutation_rate:
                chrom += np.random.normal(scale=0.1, size=len(chrom))
        return offspring

import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import norm

# 加载数据集并指定列名
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('ratings_Musical_Instruments.csv', header=None, names=column_names)

# 创建用户 ID 和项目 ID 映射
user_ids = data['user_id'].unique()
user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
item_ids = data['item_id'].unique()
item_id_map = {item_id: i for i, item_id in enumerate(item_ids)}

# 将用户 ID 和项目 ID 映射到编码
data['user_id'] = data['user_id'].apply(lambda x: user_id_map[x])
data['item_id'] = data['item_id'].apply(lambda x: item_id_map[x])

# 计算用户数和物品数
n_users = data['user_id'].nunique()
n_items = data['item_id'].nunique()

# 将评分转换为稀疏矩阵
train_data = coo_matrix((data['rating'], (data['user_id'], data['item_id'])))
train_norms = norm(train_data, axis=1)
train_norms[train_norms == 0] = 1
train_data = csr_matrix(train_data.multiply(1 / train_norms[:, None]))

# 划分训练集和测试集
test_size = 10000
test_idx = np.random.choice(np.arange(len(data)), size=test_size, replace=False)
train_idx = np.setdiff1d(np.arange(len(data)), test_idx)
X_train, y_train = train_data, data.iloc[train_idx, -1]
X_test, y_test = csr_matrix(train_data[test_idx]), data.iloc[test_idx, -1]

# 检查 test_idx 是否包含超出稀疏矩阵大小的索引
if np.max(X_test.nonzero()[0]) >= X_train.shape[0] or np.max(X_test.nonzero()[1]) >= X_train.shape[1]:
    mask = np.logical_and(X_test.nonzero()[0] < X_train.shape[0], X_test.nonzero()[1] < X_train.shape[1])
    X_test = X_test[mask]
    y_test = y_test[mask]

# 初始化遗传算法
ga = GeneticAlgorithm(n_users, n_items)

# 运行遗传算法并输出最佳个体适应度值和向量矩阵
best_fitness, (user_vectors, item_vectors) = ga.evolve(X_train, X_test, y_test)
print(f'Best fitness value: {best_fitness:.4f}')

# 使用向量矩阵进行预测并计算测试集上的 RMSE
user_norms = norm(user_vectors, axis=1)
user_norms[user_norms == 0] = 1
item_norms = norm(item_vectors, axis=1)
item_norms[item_norms == 0] = 1
user_vectors = csr_matrix(user_vectors.multiply(1 / user_norms[:, None]))
item_vectors = csr_matrix(item_vectors.multiply(1 / item_norms[:, None]))
pred_scores = user_vectors @ item_vectors.T
rmse = np.sqrt(mean_squared_error(y_test, pred_scores[test_idx]))
print(f'Test RMSE: {rmse:.4f}')


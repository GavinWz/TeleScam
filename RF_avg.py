import json

import joblib
import numpy as np
import pandas as pd
import csv

import sklearn.utils
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有列

def one_hot(column, len, df):
    '''
    :param column: 列名
    :param len: 列表长度 + 1，也就是文件行数
    '''
    # 从文件中读出城市名
    file = open('class/' + column + '.txt', 'r', encoding='utf-8')
    # 创建one-hot向量字典
    dic = {}
    idx = 0
    for line in file:  # 每一行表示一个城市名
        # print(line[:-1])
        lst = [0] * len
        lst[idx] = 1  # 每个城市一位
        dic[line[:-1]] = lst
        idx += 1
    other = [0] * len
    other[-1] = 1

    df[column] = df[column].apply(lambda x: dic[x] if x in dic else other)

# 生成序号编码
def set_index(column, df):
    file = open('class/' + column + '.txt', 'r', encoding='utf-8')
    # 创建one-hot向量字典
    dic = {}
    idx = 0
    for line in file:
        dic[line[:-1]] = idx
        idx += 1
    other = idx
    df[column] = df[column].apply(lambda x: dic[x] if x in dic else other)

def user_data(path):
    df = pd.read_csv(path)
    types = {'phone_no_m': 'str',
             'city_name': 'str',
             'county_name': 'str',
             'idcard_cnt': 'float16',
             'arpu_201908': 'float16',
             'arpu_201909': 'float16',
             'arpu_201910': 'float16',
             'arpu_201911': 'float16',
             'arpu_201912': 'float16',
             'arpu_202001': 'float16',
             'arpu_202002': 'float16',
             'arpu_202003': 'float16',
             'label': 'float16'}

    # 设置DataFrame类型
    df = df.astype(types)

    # 将所有arpu合并成一列
    df = df.melt(id_vars=['phone_no_m', 'city_name', 'county_name', 'idcard_cnt', 'label'],
                 value_vars=['arpu_201908', 'arpu_201909', 'arpu_201910', 'arpu_201911',
                             'arpu_201912', 'arpu_202001', 'arpu_202002', 'arpu_202003'],
                 var_name='arpu_origin', value_name='arpu'
                 )
    df = df.drop('arpu_origin', axis=1)

    # 对字符数据做one-hot
    one_hot('city_name', 24, df)
    one_hot('county_name', 183, df)

    # 将city_name和county_name的one-hot vector拆成每个元素一列
    city_name_np = np.array(df['city_name'].tolist())
    new_columns = ['city_name' + str(i) for i in range(city_name_np.shape[1])]
    city_name_df = pd.DataFrame(city_name_np, columns=new_columns, dtype=np.float16)

    county_name_np = np.array(df['county_name'].tolist())
    new_columns = ['county_name' + str(i) for i in range(county_name_np.shape[1])]
    county_name_df = pd.DataFrame(county_name_np, columns=new_columns, dtype=np.float16)

    # 删除掉原来的'city_name', 'county_name'向量
    df = df.drop(['city_name', 'county_name'], axis=1)
    # 将label先删掉重新添加，以使其到DataFrame的最后一列
    label_df = df.pop('label')
    df = pd.concat([df, city_name_df, county_name_df, label_df], axis=1)
    df.replace('nan', '', inplace=True)
    df.dropna(inplace=True)

    x = df.iloc[:, :-1]
    y = df['label']
    # 划分训练集和测试集
    x, y = sklearn.utils.shuffle(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print('User data loaded successfully!!')
    return x_train, x_test, y_train, y_test

    # # 将城市名存入文件
    # citys = set(df['county_name'].tolist())
    # print('nan' in citys)
    # file = open('class/county_name.txt', 'w', encoding='utf-8')
    # file.write('\n')
    # for city in citys:
    #     file.write(city + '\n')
    #
    # # 从文件中读出城市名
    # file = open('class/county_name.txt', 'r', encoding='utf-8')
    # lst = []
    # for line in file:
    #     lst.append(line[:-1])


def app_data(path):
    df = pd.read_csv(path)

    types = {'phone_no_m': 'str',
             'busi_name': 'str',
             'flow': 'float32',
             'month_id': 'str',
             'label': 'float16'
             }

    # 设置DataFrame类型
    df = df.astype(types)
    df['busi_name'] = df['busi_name'].replace('nan', '')
    df['flow'] = df['flow'].fillna(0)
    df['month_id'] = df['month_id'].fillna('').str.rstrip('\r')
    df['month_id'] = pd.to_datetime(df['month_id'], format='%Y-%m')
    df['year'] = df['month_id'].dt.year.fillna(0).astype(np.float16)
    df['month'] = df['month_id'].dt.month.fillna(0).astype(np.float16)
    df['label'] = df['label'].dropna().astype(np.float16)

    # 对字符数据做序列化
    set_index('busi_name', df)
    # 将输入的所有数据合并成一个列表
    x = df[['phone_no_m', 'busi_name', 'flow', 'year', 'month']]
    y = df['label']

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print('App data loaded successfully!!')
    return x_train, x_test, y_train, y_test

    # # 将软件名存入文件
    # citys = set(df['busi_name'].tolist())
    # print('nan' in citys)
    # file = open('class/busi_name.txt', 'w', encoding='utf-8')
    # file.write('\n')
    # for city in citys:
    #     file.write(city + '\n')

    # # 从文件中读出软件名
    # file = open('class/busi_name.txt', 'r', encoding='utf-8')
    # lst = []
    # for line in file:
    #     lst.append(line[:-1])
    # print(lst)


def sms_data(path):
    df = pd.read_csv(path)
    types = {'phone_no_m': 'str',
             'opposite_no_m': 'str',
             'calltype_id': 'float16',
             'request_datetime': 'str',
             'label': 'float16'
             }
    # 设置DataFrame类型
    df = df.astype(types)
    df['calltype_id'] = df['calltype_id'].fillna(0)
    df['request_datetime'] = df['request_datetime'].fillna('')
    df['request_datetime'] = pd.to_datetime(df['request_datetime'])
    df['year'] = df['request_datetime'].dt.year.fillna(0)
    df['month'] = df['request_datetime'].dt.month.fillna(0)
    df['day'] = df['request_datetime'].dt.day.fillna(0)
    df['hour'] = df['request_datetime'].dt.hour.fillna(0)
    # df['label'] = df['label'].dropna()

    x = df[['phone_no_m', 'calltype_id', 'year', 'month', 'day', 'hour']]
    y = df['label']
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print('Sms data loaded successfully!!')
    return x_train, x_test, y_train, y_test


def voc_data(path):
    df = pd.read_csv(path)
    types = {'phone_no_m': 'str',
             'opposite_no_m': 'str',
             'calltype_id': 'short',
             'start_datetime': 'str',
             'call_dur': 'float16',
             'city_name': 'str',
             'county_name': 'str',
             'imei_m': 'str',
             'label': 'short'
             }
    # 设置DataFrame类型
    df = df.astype(types)
    # print(df.dtypes)

    # 处理日期类型
    df['start_datetime'] = df['start_datetime'].fillna('')
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df['year'] = df['start_datetime'].dt.year.fillna(0).astype(np.float16)
    df['month'] = df['start_datetime'].dt.month.fillna(0).astype(np.float16)
    df['day'] = df['start_datetime'].dt.day.fillna(0).astype(np.float16)
    df['hour'] = df['start_datetime'].dt.hour.fillna(0).astype(np.float16)
    df.drop('start_datetime', axis=1, inplace=True)

    set_index('opposite_no_m', df)

    # 处理city_name和county_name
    one_hot('city_name', 24, df)
    one_hot('county_name', 183, df)

    # 将city_name和county_name的one-hot vector拆成每个元素一列
    city_name_np = np.array(df['city_name'].tolist())
    new_columns = ['city_name' + str(i) for i in range(city_name_np.shape[1])]
    city_name_df = pd.DataFrame(city_name_np, columns=new_columns)

    county_name_np = np.array(df['county_name'].tolist())
    new_columns = ['county_name' + str(i) for i in range(county_name_np.shape[1])]
    county_name_df = pd.DataFrame(county_name_np, columns=new_columns)

    # 删除掉原来的'city_name', 'county_name'向量
    df = df.drop(['city_name', 'county_name', 'imei_m'], axis=1)

    # 将label先删掉重新添加，以使其到DataFrame的最后一列
    label_df = df.pop('label')
    df = pd.concat([df, city_name_df, county_name_df, label_df], axis=1)

    # 处理空值
    df.replace('nan', '', inplace=True)
    # df.dropna(inplace=True)
    x = df.iloc[:, :-1]
    y = df['label'].astype(np.float16)
    # print(x.)
    # 划分训练集和测试集
    x, y = sklearn.utils.shuffle(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # # 将imei_m做哈希处理
    # hasher = FeatureHasher(n_features=200, input_type='string')
    # hashed_features = hasher.transform([[imei] for imei in df['imei_m'].tolist()])
    # # print(hashed_features)
    #
    # # 将哈希后的特征转换为DataFrame
    # hashed_features_df = pd.DataFrame(hashed_features.toarray())
    # df = pd.concat([hashed_features_df, df], axis=1)
    # print(df)
    # hashed_features_df.to_csv('imei_hash.csv')
    '''
        后续将训练集中所有诈骗电话的imei存到字典中，训练时只要碰到字典中的imei即判断为诈骗电话
    '''
    print('Voc data loaded successfully!!')
    return x_train, x_test, y_train, y_test

def user_train(x_train, x_test, y_train, y_test):

    # 创建随机森林分类器实例
    # clf = RandomForestClassifier(n_estimators=100, random_state=0, verbose=2, n_jobs=-1)
    # clf = knn = KNeighborsClassifier(n_neighbors=3)
    clf = GradientBoostingClassifier(n_estimators=128, learning_rate=0.2, max_depth=16, random_state=42)
    # 训练模型
    clf.fit(x_train, y_train)

    # 预测测试集
    y_pred = clf.predict(x_test)
    joblib.dump(clf, 'rf_model/user_crf.pkl')
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"user模型准确率: {accuracy:.5f}")
    return clf, y_pred


def app_train(x_train, x_test, y_train, y_test):
    # # 对字符数据做Hash (内存不够，能用自己实现的one-hot函数实现)
    # hasher = FeatureHasher(n_features=500)
    # data_to_hash = [{'busi_name': val} for val in df['busi_name']]
    # hashed_features = hasher.fit_transform(data_to_hash)
    # busi_name_df = pd.DataFrame(hashed_features.toarray()).astype(np.float16)
    # df.drop(['phone_no_m', 'busi_name'], axis=1, inplace=True)
    # df = pd.concat([busi_name_df, df], axis=1)
    # print(df)
    #
    # # # 将输入的所有数据合并成一个列表
    # x = df.iloc[:, :-1].to_numpy()
    # y = df.iloc[:, -1].to_numpy()

    # 创建分类器实例
    # clf = RandomForestClassifier(n_estimators=8, random_state=0, verbose=2, n_jobs=-1)
    # clf = SVC(kernel='linear')
    clf = KNeighborsClassifier(n_neighbors=8)
    # 训练模型
    clf.fit(x_train, y_train)
    # 预测测试集
    y_pred = clf.predict(x_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"app模型准确率: {accuracy:.5f}")
    joblib.dump(clf, 'rf_model/app_crf.pkl')
    return clf, y_pred


def sms_train(x_train, x_test, y_train, y_test):

    # 创建随机森林分类器实例
    # clf = RandomForestClassifier(n_estimators=8, random_state=0, verbose=2, n_jobs=-1)
    clf = KNeighborsClassifier(n_neighbors=8)
    # 训练模型
    clf.fit(x_train, y_train)
    # 预测测试集
    y_pred = clf.predict(x_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"sms模型准确率: {accuracy:.5f}")
    joblib.dump(clf, 'rf_model/sms_crf.pkl')
    return clf, y_pred


def voc_train(x_train, x_test, y_train, y_test):
    # 训练阶段忽略imei
    X_train = x_train.drop('imei_m', axis=1)
    X_test = x_test.drop('imei_m', axis=1)

    # 创建随机森林分类器实例
    clf = RandomForestClassifier(n_estimators=8, random_state=0, verbose=2, n_jobs=-1)
    # clf = SVC(verbose=True)
    # clf = joblib.load('rf_model/voc_crf.pkl')
    # 训练模型
    clf.fit(X_train, y_train)

    # 预测测试集
    y_pred = clf.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"voc模型准确率: {accuracy:.6f}")
    joblib.dump(clf, 'rf_model/voc_crf.pkl')
    return clf, y_pred

# def model_test(model, x_test, y_test):


def load_models(path):
    user_model = joblib.load(path + '/user_crf.pkl')
    print('User model loaded successfully!!')
    app_model = joblib.load(path + '/app_crf.pkl')
    print('App model loaded successfully!!')
    sms_model = joblib.load(path + '/sms_crf.pkl')
    print('Sms model loaded successfully!!')
    voc_model = joblib.load(path + '/voc_crf.pkl')
    print('Voc model loaded successfully!!')
    return user_model, app_model, sms_model, voc_model

def model_test(model, x_test, y):
    '''
    计算每个文件中每个用户预测结果的均值，保存到字典中，phone_no_m:
    '''
    dic = {}
    # print(x_test.columns)
    for i in range(len(y)):
        phone = x_test.iloc[i, 0]
        if phone not in dic:
            dic[phone] = [y[i]]
        else:
            dic[phone].append(y[i])
    for key in dic:
        avg = sum(dic[key]) / len(dic[key])
        dic[key] = avg
    return dic


# 加载数据
user_x_train, user_x_test, user_y_train, user_y_test = user_data('dataset/train/train_user.csv')
app_x_train, app_x_test, app_y_train, app_y_test = app_data('dataset/train/train_app.csv')
sms_x_train, sms_x_test, sms_y_train, sms_y_test = sms_data('dataset/train/train_sms.csv')
voc_x_train, voc_x_test, voc_y_train, voc_y_test = voc_data('dataset/train/voc.csv')


def test0():
    # 加载模型
    user_model, app_model, sms_model, voc_model = load_models('rf_model')

    user_y = user_model.predict(user_x_test.iloc[:, 1:])
    app_y = app_model.predict(app_x_test.iloc[:, 1:])
    sms_y = sms_model.predict(sms_x_test.iloc[:, 1:])
    voc_y = voc_model.predict(voc_x_test.iloc[:, 1:])

    user_dic = model_test(user_model, user_x_test, user_y)
    app_dic = model_test(app_model, app_x_test, app_y)
    sms_dic = model_test(sms_model, sms_x_test, sms_y)
    voc_dic = model_test(voc_model, voc_x_test, voc_y)

    # 临时
    import json
    with open('avg_user_dic.json', 'w') as file:
        json.dump(user_dic, file, indent=4)
    with open('avg_app_dic.json', 'w') as file:
        json.dump(app_dic, file, indent=4)
    with open('avg_sms_dic.json', 'w') as file:
        json.dump(sms_dic, file, indent=4)
    with open('avg_voc_dic.json', 'w') as file:
        json.dump(voc_dic, file, indent=4)
    #####

    print(len(user_dic), len(user_dic), len(user_dic), len(user_dic))

    '''
        合并思想：根据模型正确率user 91.7%, app 81%, sms 78%, voc 71%
        1. 先将几个表中出现的所有user合并到同一个集合中
        2. user正确率最高，所以如果user有预测值就取user的值
        3. 如果某个用户在user的预测结果中没有出现，判断其他三个模型
    '''

    dic_final = {}  # 保存每个用户最终的预测结果
    all_users = set(user_dic.keys()) | set(app_dic.keys()) | set(sms_dic.keys()) | set(voc_dic.keys())
    print(len(all_users))
    for user in all_users:
        scores = []
        if user in user_dic:
            scores.append(user_dic[user])
        if user in app_dic:
            scores.append(app_dic[user])
        if user in sms_dic:
            scores.append(sms_dic[user])
        if user in voc_dic:
            scores.append(voc_dic[user])
        avg = sum(scores) / len(scores)
        avg = 1 if avg > 0.5 else 0
        dic_final[user] = avg

    # 统计所有的用户的label，不要重复的数据
    label_dic = {}
    for i in range(len(user_x_test)):
        phone = user_x_test.iloc[i, 0]
        if phone not in label_dic:
            label_dic[phone] = user_y_test.iloc[i]
    for i in range(len(app_x_test)):
        phone = app_x_test.iloc[i, 0]
        if phone not in label_dic:
            label_dic[phone] = app_y_test.iloc[i]
    for i in range(len(sms_x_test)):
        phone = sms_x_test.iloc[i, 0]
        if phone not in label_dic:
            label_dic[phone] = sms_y_test.iloc[i]
    for i in range(len(voc_x_test)):
        phone = voc_x_test.iloc[i, 0]
        if phone not in label_dic:
            label_dic[phone] = voc_y_test.iloc[i]

    # 最后的测试
    correct_cnt = 0
    for phone in dic_final:
        if dic_final[phone] == label_dic[phone]:
            correct_cnt += 1

    print(correct_cnt / len(dic_final))

def test():
    # 临时
    import json
    with open('avg_user_dic.json', 'r') as file:
        user_dic = json.load(file)
    with open('avg_app_dic.json', 'r') as file:
        app_dic = json.load(file)
    with open('avg_sms_dic.json', 'r') as file:
        sms_dic = json.load(file)
    with open('avg_voc_dic.json', 'r') as file:
        voc_dic = json.load(file)
    #####

    print(len(user_dic), len(user_dic), len(user_dic), len(user_dic))

    dic_final = {}  # 保存每个用户最终的预测结果
    all_users = set(user_dic.keys()) | set(app_dic.keys()) | set(sms_dic.keys()) | set(voc_dic.keys())
    print(len(all_users))
    for user in all_users:
        scores = []
        if user in user_dic:
            scores.append(user_dic[user])
        if user in app_dic:
            scores.append(app_dic[user])
        if user in sms_dic:
            scores.append(sms_dic[user])
        if user in voc_dic:
            scores.append(voc_dic[user])
        avg = sum(scores) / len(scores)
        avg = 1 if avg > 0.5 else 0
        dic_final[user] = avg

    # 统计所有的用户的label，不要重复的数据
    label_dic = {}
    for i in range(len(user_x_test)):
        phone = user_x_test.iloc[i, 0]
        if phone not in label_dic:
            label_dic[phone] = user_y_test.iloc[i]
    for i in range(len(app_x_test)):
        phone = app_x_test.iloc[i, 0]
        if phone not in label_dic:
            label_dic[phone] = app_y_test.iloc[i]
    for i in range(len(sms_x_test)):
        phone = sms_x_test.iloc[i, 0]
        if phone not in label_dic:
            label_dic[phone] = sms_y_test.iloc[i]
    for i in range(len(voc_x_test)):
        phone = voc_x_test.iloc[i, 0]
        if phone not in label_dic:
            label_dic[phone] = voc_y_test.iloc[i]

    # 最后的测试
    correct_cnt = 0
    for phone in dic_final:
        if dic_final[phone] == label_dic[phone]:
            correct_cnt += 1

    print(correct_cnt / len(dic_final))

if __name__ == '__main__':

    # 模型训练
    # user_model, user_y = user_train(user_x_train.iloc[:, 1:], user_x_test.iloc[:, 1:], user_y_train, user_y_test)
    # app_model, app_y = app_train(app_x_train.iloc[:, 1:], app_x_test.iloc[:, 1:], app_y_train, app_y_test)
    # sms_model, sms_y = sms_train(sms_x_train.iloc[:, 1:], sms_x_test.iloc[:, 1:], sms_y_train, sms_y_test)
    # voc_model, voc_y = voc_train(voc_x_train.iloc[:, 1:], voc_x_test.iloc[:, 1:], voc_y_train, voc_y_test)
    test()


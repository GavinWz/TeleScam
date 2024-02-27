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
             'idcard_cnt': 'float32',
             'arpu_202004': 'float32'
             }
    # 设置DataFrame类型
    df = df.astype(types)
    df.rename(columns={'arpu_202004': 'arpu'}, inplace=True)

    df.dropna(subset=['phone_no_m'], inplace=True)
    df['city_name'].replace('nan', '', inplace=True)
    df['county_name'].replace('nan', '', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['idcard_cnt'].fillna(value=df['idcard_cnt'].mean(), inplace=True)
    df['arpu'].fillna(value=df['arpu'].mean(), inplace=True)

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
    df = pd.concat([df, city_name_df, county_name_df], axis=1)
    print('User data loaded successfully!!')
    return df

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
             'month_id': 'str'
             }

    # 设置DataFrame类型
    df = df.astype(types)
    df['busi_name'] = df['busi_name'].replace('nan', '')
    df['flow'] = df['flow'].fillna(0)
    df['month_id'] = df['month_id'].fillna('').str.rstrip('\r')
    df['month_id'] = pd.to_datetime(df['month_id'], format='%Y-%m')
    df['year'] = df['month_id'].dt.year.fillna(0).astype(np.float16)
    df['month'] = df['month_id'].dt.month.fillna(0).astype(np.float16)

    # 对字符数据做序列化
    set_index('busi_name', df)
    # 将输入的所有数据合并成一个列表
    x = df[['phone_no_m', 'busi_name', 'flow', 'year', 'month']]

    print('App data loaded successfully!!')
    return x

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
             'request_datetime': 'str'
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

    x = df[['phone_no_m', 'calltype_id', 'year', 'month', 'day', 'hour']]
    # 划分训练集和测试集
    print('Sms data loaded successfully!!')
    return x


def voc_data(path):
    df = pd.read_csv(path)
    types = {'phone_no_m': 'str',
             'opposite_no_m': 'str',
             'calltype_id': 'short',
             'start_datetime': 'str',
             'call_dur': 'float16',
             'city_name': 'str',
             'county_name': 'str',
             'imei_m': 'str'
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

    # 处理空值
    df.replace('nan', '', inplace=True)

    # 处理opposite_no_m、city_name和county_name
    set_index('opposite_no_m', df)
    one_hot('city_name', 24, df)
    one_hot('county_name', 183, df)

    # 将city_name和county_name的one-hot vector拆成每个元素一列
    city_name_np = np.array(df['city_name'].tolist())
    new_columns = ['city_name' + str(i) for i in range(city_name_np.shape[1])]
    city_name_df = pd.DataFrame(city_name_np, columns=new_columns)

    county_name_np = np.array(df['county_name'].tolist())
    new_columns = ['county_name' + str(i) for i in range(county_name_np.shape[1])]
    county_name_df = pd.DataFrame(county_name_np, columns=new_columns)

    # 删除掉原来的'city_name', 'county_name'向量，以及imei_m字段
    df = df.drop(['city_name', 'county_name', 'imei_m'], axis=1)

    df = pd.concat([df, city_name_df, county_name_df], axis=1)

    print('Voc data loaded successfully!!')
    return df

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
        dic[key] = 1 if avg > 0.6 else 0
    return dic


# 加载数据
user_x_test = user_data('dataset/test/test_user.csv')
app_x_test = app_data('dataset/test/test_app.csv')
sms_x_test = sms_data('dataset/test/test_sms.csv')
voc_x_test = voc_data('dataset/test/test_voc.csv')


def inference():
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

    print(len(user_dic), len(app_dic), len(sms_dic), len(voc_dic))
    '''
        合并思想：根据模型正确率user 91.7%, app 81%, sms 78%, voc 71%
        1. 先将几个表中出现的所有user合并到同一个集合中
        2. user正确率最高，所以如果其他三个相同时才取他们的值，否则就取user的值
        3. 如果某个用户在user的预测结果中没有出现，则按app的结果来
        4. 如果在app的预测结果中也没有出现，则按sms的结果来，还没有的话一定在voc里，按voc的结果来。
    '''

    dic_final = {}  # 保存每个用户最终的预测结果
    all_users = set(user_dic.keys()) | set(app_dic.keys()) | set(sms_dic.keys()) | set(voc_dic.keys())
    print(len(all_users))
    for user in all_users:
        posi = 0
        nega = 0
        if user in app_dic:
            if app_dic[user] == 1:
                posi += 1
            else:
                nega += 1
        if user in sms_dic:
            if sms_dic[user] == 1:
                posi += 1
            else:
                nega += 1
        if user in voc_dic:
            if voc_dic[user] == 1:
                posi += 1
            else:
                nega += 1

        # 如果user之外的三个模型的预测结果相同，则将最终结果设置为此结果，否则取user模型的结果做为最终结果
        if user in user_dic:
            # if posi == 3:
            #     dic_final[user] = 1
            # elif nega == 3:
            #     dic_final[user] = 0
            # else:
            dic_final[user] = user_dic[user]
        # 如果不在user表中，则判断预测为正样本的个数是否大于负样本的个数
        else:
            if posi - nega > 0:
                dic_final[user] = 1
            else:
                dic_final[user] = 0
    print(sum(dic_final.values()))
    result = pd.DataFrame(list(dic_final.items()), columns=['phone_no_m', 'label'])
    result.to_csv('final_result.csv')




if __name__ == '__main__':

    inference()


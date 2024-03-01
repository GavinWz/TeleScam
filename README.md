# 反电信诈骗研究 -- 机器学习方法 

完整训练数据和模型训练结果：https://www.alipan.com/s/TLdwdFDmQnV

## 数据集分析和预处理

数据集中除了user表其他表都没有label，将四个训练表导入mysql数据库中，通过查询的方法从user表中查找对应的label，添加其他三个表中到对应的数据中

## 数据加载、预处理和特征工程 

`user_data`：读取train_user数据，将所有arpu合成一列，生成city_name.txt、county_name.txt到class文件夹下，表示出现过得城市名称和公司名称，空数据置为空字符串。两个文件中第一行均为空字符串，便于训练时将空字符串编码为0，然后将训练数据中的空数据均设置为空字符串，进行one-hot操作

`app_data`：读取train_app数据，将出现过的app名称保存到busi_name.txt中，第一行也是空字符串，便于置为0。将年月拆成年和月两列。

## 模型训练

`app_train`：三次尝试
1. 使用one-hot函数对busi_name做独热编码，失败。  
    原因：busi_name一共4996类，3280000+条数据，做one-hot对内存消耗巨大。
2. 使用set_index函数对busi_name做序列化。  
    结果：78.7%的正确率
3. 使用sklearn中的FeatureHasher函数对busi_name做hash操作。  
    结果：76% (n_features=100)  77% (n_features=500)

结论：随机森林中决策树分类器的个数对模型的训练没有什么影响。综上，取第二种方法。

`sms_data`：读取train_sms数据，忽略opposite列，将日期类型request_datetime拆分为年月日时四列。

`voc_data`：读取train_voc数据，将训练集中所有诈骗电话的imei存到字典中，训练时只要碰到字典中的imei即判断为诈骗电话。训练时的DataFrame不保存imei列，读取文件city_name.txt和county_name.txt中的数据，分别存入两个列表，利用这两个列表做one-hot，最后与其余数据拼接。经统计发现，测试集的imei在训练集中出现的比例只有1%，而且imei是没有语义的特征，所以在训练时直接删掉.
calltypeid和call_dur经检查不存在空值

各个模型单独的准确率分别是：   
    
| model      | acc   |
|------------|-------|
| user model | 92.1% |  
| app model  | 84.4% |
| sms model  | 80.2% |  
| voc model  | 79.3% |

最终模型的正确率: 83.3%

| Strategy | Accuracy |
|----------|----------|
| Voting   | 91.7%    |
| Average  | 83.45%   |

Accuracy: 0.9179495578119882
Precision: 0.9603024574669187
Recall: 0.7767584097859327
R2 Score: 0.6237506403714577

## 遇到的挑战

1. 数据集是多个csv文件，只有user表里有label，但是每个文件中都有phone_no_m，两次尝试：
   * 导入mysql数据库，尝试连接多个表，将所有信息保存到同一个表中。  
    失败：voc和sms表有上六百万行数据，合并极其耗时，跑了两天没有合并出来，遂放弃。
   * 表的合并开销极大，考虑尝试通过phone_no_m从user表取出对应label放入其他三个表，如果某个phone_no_m没有在user表中找到label，则直接删除该数据，没有label的数据是没有意义的。也是使用mysql实现，在表中添加label列中导出csv文件。
2. 模型太大，只有16G内存，不能一起跑，将训练任务拆成四份单独训练，将每一个表的训练结果保存下来。然后修改代码，跳过训练过程直接加载模型用于测试。
3. 向mysql中导入train_voc时，没有调字符集，导致中文数据没有导入成功，训练时候没有发现，得到的准确率很低
4. 将诈骗电话的imei记录下来，如果测试集出现被记录的imei直接被认定为诈骗电话。（淘汰，test_voc中一共2656个用户，只有24个用户出现在记录中，所占比例太少）





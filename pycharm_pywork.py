import pandas as pd
from sklearn import svm


# 导入数据、划分特征属性集和标签集
labeled_images = pd.read_csv('train.csv')
train_images = labeled_images.drop('label', axis=1)
train_labels = labeled_images.label
# print(labeled_images.head())
# print(train_images.head())
# print(train_labels.head())


#  模型优化，特征0/1化（有值的都设为1，没值的保持为0）
train_images[train_images>0]=1

# 构建模型、训练模型
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf

# 预测测试集的结果
test_data=pd.read_csv('test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data)
results

# 输出标准提交文件
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('submission_results.csv', header=True)

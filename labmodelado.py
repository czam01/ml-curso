#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python
# coding: utf-8

bucket = 'czambrano-sagemaker' 
prefix = 'sagemaker/xgboost_credit_risk'


# In[6]:


import boto3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sagemaker
import csv
from sagemaker import get_execution_role
from sagemaker.predictor import csv_serializer


# In[7]:


bucket


# In[8]:


prefix


# In[9]:


role = get_execution_role() 


# In[10]:


role


# In[11]:


get_ipython().system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls')


# In[12]:


dataset = pd.read_excel('default of credit card clients.xls')


# In[13]:


pd.set_option('display.max_rows', 8)


# In[14]:


pd.set_option('display.max_columns', 15)


# In[15]:


dataset


# In[16]:


dataset = dataset.drop('Unnamed: 0', axis=1)


# In[17]:


dataset


# In[18]:


dataset = pd.concat([dataset['Y'], dataset.drop(['Y'], axis=1)], axis=1)


# In[19]:


dataset


# In[20]:


train_data, validation_data, test_data = np.split(dataset.sample(frac=1, random_state=1729), [int(0.7 * len(dataset)), int(0.9 * len(dataset))])


# In[21]:


train_data.to_csv('train.csv', header=False, index=False)


# In[22]:


train_data


# In[23]:


validation_data.to_csv('validation.csv', header=False, index=False)


# In[24]:


validation_data


# In[25]:


boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('validation.csv')
s3_input_train = TrainingInput(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = TrainingInput(s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='csv')


# In[26]:


boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')


# In[27]:


boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('validation.csv')


# In[28]:


s3_input_train = TrainingInput(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')


# In[29]:


from sagemaker.inputs import TrainingInput


# In[30]:


s3_input_train = TrainingInput(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = TrainingInput(s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='csv')


# In[31]:


s3_input_train


# In[32]:


s3_input_validation


# In[33]:


containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}


# In[34]:


containers


# In[35]:


sess = sagemaker.Session()


# In[36]:


xgb = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                    role, 
                                    instance_count=1, 
                                    instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)


# In[37]:


xbg


# In[38]:


xgb.set_hyperparameters(eta=0.1, objective='binary:logistic', num_round=25) 


# In[39]:


xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})


# In[42]:


xgb_predictor = xgb.deploy(
	initial_instance_count = 1,
	instance_type = 'ml.m4.xlarge',
	serializer = CSVSerializer())


# In[41]:


from sagemaker.serializers import CSVSerializer


# In[43]:


def predict(data, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, xgb_predictor.predict(array).decode('utf-8')])

    return np.fromstring(predictions[1:], sep=',')

predictions = predict(test_data.to_numpy()[:,1:])


# In[44]:


predictions


# In[ ]:





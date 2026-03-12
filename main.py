
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def read_parquet_file(file_name):
    df = pd.read_parquet(file_name, engine="fastparquet")
    print(file_name)
    print(df.head())
    print('info')
    print(df.info())
    return df

sales = read_parquet_file('20260218_144523_sales_data.parquet')
holidays = read_parquet_file('20260218_144523_holidays.parquet')
stores = read_parquet_file('20260218_144523_stores.parquet')
weather = read_parquet_file('20260218_144523_weather.parquet')

sales['date'].min()
sales['date'].max()

sales['month'] = sales['date'].dt.month
sales['weekday'] = sales['date'].dt.weekday
sales['weekday'] = sales['weekday'] + 1
#remove the two strange categories. there is only one line for each
sales = sales[(sales['category_name'] != 'Brotwaage') & (sales['category_name'] != 'Angebot Gastro')]
#sample check if the data is aggregated
print(sales.loc[(sales['item_id'] == 139) & (sales['store_id'] == 0)])
#there are some price missing
print(sales.loc[sales['price'].isna()][['date', 'category_name', 'item_id', 'price']])
sales_agg1 = sales[['category_name', 'sold_quantity']].groupby(['category_name']).sum()
sales_agg1 = sales_agg1.reset_index().sort_values('sold_quantity', ascending = False)
# plot the total sold quantities of the top 10 bestsellers
plt.figure(figsize = (12,8))
sns.barplot(x = 'category_name', y = 'sold_quantity', data = sales_agg1.iloc[:10])
plt.xlabel('category')
plt.ylabel('sold quantity')
plt.title('Top 10 categories')
plt.savefig('top_10_cat.png')
#plt.show()

# plot the sold quantities of the top 10 categoties by weekdays
top10_categories = sales_agg1['category_name'].iloc[:10]
other_categories = sales_agg1['category_name'].iloc[10:]

sales_agg2 = sales[['category_name', 'weekday', 'sold_quantity']].groupby(['category_name', 'weekday' ]).sum()
sales_agg2 = sales_agg2.reset_index()
sales_agg2_top = sales_agg2[sales_agg2['category_name'].isin(top10_categories)]
plt.figure(figsize = (12,8))
sns.barplot(x = 'category_name', y = 'sold_quantity', hue = 'weekday', data = sales_agg2_top)
plt.xlabel('category')
plt.ylabel('sold quantity')
plt.title('Top 10 categories by weekday')
plt.savefig('top_10_cat_weekday.png')
#plt.show()

sales_agg2_other = sales_agg2[sales_agg2['category_name'].isin(other_categories)]
plt.figure(figsize = (12,8))
sns.barplot(x = 'category_name', y = 'sold_quantity', hue = 'weekday', data = sales_agg2_other)
plt.xlabel('category')
plt.ylabel('sold quantity')
plt.title('other categories by weekday')
plt.savefig('other_cat_weekday.png')
#plt.show()
# plot the sold quantities of the top 10 categoties by weekdays
#categories = sales['category_name'].unique()
#categories = categories.astype('str')
#categories = np.char.replace(categories, '/', '_')
#sales_agg3 = sales[['category_name', 'month', 'sold_quantity']].groupby(['category_name', 'month']).sum()
#sales_agg3 = sales_agg3.reset_index()
#for category in categories:
#    picture_name = category + '_by_month.png'
#    plt.figure(figsize=(12, 8))
#    sns.barplot(x='category_name',
#                y='sold_quantity',
#                hue='month', data=sales_agg3[sales_agg3['category_name']==category])
#    plt.xlabel('category')
#    plt.ylabel('sold quantity')
#    plt.title('')
#    plt.savefig(picture_name)
sales_agg3 = sales[['category_name', 'month', 'sold_quantity']].groupby(['category_name', 'month']).sum()
sales_agg3 = sales_agg3.reset_index()
sales_agg3_top = sales_agg3[sales_agg3['category_name'].isin(top10_categories)]
plt.figure(figsize = (12,8))
sns.barplot(x = 'category_name', y = 'sold_quantity', hue = 'month', data = sales_agg3_top)
plt.xlabel('category')
plt.ylabel('sold quantity')
plt.title('Top 10 categories by month')
plt.savefig('top_10_cat_month.png')
#plt.show()

sales_agg3_other = sales_agg3[sales_agg3['category_name'].isin(other_categories)]
plt.figure(figsize = (12,8))
sns.barplot(x = 'category_name', y = 'sold_quantity', hue = 'month', data = sales_agg3_other)
plt.xlabel('category')
plt.ylabel('sold quantity')
plt.title('Other categories by month')
plt.savefig('other_cat_month.png')
#plt.show()


categories = sales['category_name'].unique()
categories = categories.astype('str')
categories = np.char.replace(categories, '/', '_')
for category in categories:
    picture_name = category + '_quant_hist.png'
    plt.figure(figsize=(12, 8))
    sns.displot(sales[sales['category_name'] == category]['sold_quantity'], kde=True, rug=True)
    plt.savefig(picture_name)

df = pd.merge(sales, holidays, left_on = 'date', right_on='date', how='left')
df = pd.merge(df, stores, left_on = 'store_id', right_on='store_id', how='left')
df = pd.merge(df, weather, left_on = 'date', right_on='date', how='left')



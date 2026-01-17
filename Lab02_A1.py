import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("A1:")
#A1
df=pd.read_excel("Lab02.xlsx",sheet_name="Purchase_data")
df=df.iloc[:,:5]
X1=df.drop(columns=['Payment (Rs)','Customer'])
y1=df["Payment (Rs)"]
print(X1)
print(y1)
rank=np.linalg.matrix_rank(X1)
print(rank)
x1_pinv=np.linalg.pinv(X1)
c=np.matmul(x1_pinv,y1)
print(x1_pinv)
print(c)

print("A2:")
#A2
df['status']=df['Payment (Rs)'].apply(lambda x:'rich' if x>200 else 'poor')
y1_status=df['status']
print(y1_status)

print("A3:")
#A3
df=pd.read_excel("Lab02.xlsx",sheet_name="IRCTC_Stock_Price")
price=df["Price"]
def manual_mean(price):
    return f"mean:{sum(price)/len(price)}"
def manual_var(price):
    mu=manual_mean(price)
    return f"variance:{sum((x-mu)**2 for x in price)/len(price)}"
package_mean=np.mean(price)
package_var=np.var(price)
def check():
    if manual_mean(price)==package_mean:
        pass
print(package_mean)
print(package_var)


def mean_wednesday(price):
    day=df["Day"]
    price_wed=[]
    for x in range(len(day)):
        if day[x]=="Wed":
            price_wed.append(price[x])
    return np.mean(price_wed)

def wed_price_mean_compare(price):
    if mean_wednesday(price) == manual_mean(price):
        return f"same(mean of wednesday:{mean_wednesday(price)}, mean of population:{manual_mean(price)})"
    else:
        return f"not same(mean of wednesday:{mean_wednesday(price)}, mean of population:{manual_mean(price)})"
    
def mean_april(price):
    month=df["Month"]
    price_april=[]
    for x in range(len(month)):
        if month[x] == "Apr":
            price_april.append(price[x])
    return np.mean(price_april)

def apr_price_mean_compare(price):
    if mean_april(price) == manual_mean(price):
        return f"same(mean of april:{mean_april(price)}, mean of population:{manual_mean(price)})"
    else:
        return f"not same(mean of april:{mean_april(price)}, mean of population:{manual_mean(price)})"
    
def prob_loss(price):
    chg=df["Chg%"]
    total_days=len(df)
    loss=list(filter(lambda x: x<0, chg)) # filtering if x<0(loss) and storing it in list
    total_loss=len(loss) # finding length of loss list
    return total_loss/total_days # calculating probability of loss 

chg=df["Chg%"]
day=df["Day"]

def profit_on_wed(price):
    profit_wed=0 # initializing profit on wednesday count as zero
    total_days=len(df)
    for x in range(len(day)):
     if day[x]=="Wed" and chg[x]>0:
        profit_wed+=1 # updating profit on wednesday count by adding 1
    return profit_wed/total_days

def profit_if_today_wed(price):
    profit_wed=0
    wed_count=0
    for x in range(len(day)):
     if day[x]=="Wed":
        wed_count+=1
        if chg[x]>0:
         profit_wed+=1
    return profit_wed/wed_count

def scatter_plot():
    day_map={
        "Mon":1,
        "Tue":2,
        "Wed":3,
        "Thu":4,
        "Fri":5
    }
    day_num=df["Day"].map(day_map)
    
    plt.figure()
    plt.scatter(day_num,df["Chg%"])
    plt.xlabel("Day of the week")
    plt.ylabel("Chg%")
    plt.xticks([1,2,3,4,5], ["Mon","Tue","Wed","Thu","Fri"])
    plt.grid(True)
    plt.show()

print("A4:")

df1=pd.read_excel("Lab02.xlsx",sheet_name="thyroid0387_UCI")
def mean_var_std():
    numeric_cols = ['age','TSH','T3','TT4','T4U','FTI','TBG']

    for col in numeric_cols:
       df1[col] = pd.to_numeric(df1[col], errors='coerce')
    return df1[numeric_cols].mean(), df1[numeric_cols].var(), df1[numeric_cols].std()

print("A5:")
# A5
def similarity_measure():
    doc1=df1.iloc[0]
    doc2=df1.iloc[1]  # 2 observation vectors i.e 1st two rows
    binary_col = [
    'on thyroxine', 'query on thyroxine', 'on antithyroid medication',
    'sick', 'pregnant', 'thyroid surgery', 'I131 treatment',
    'query hypothyroid', 'query hyperthyroid', 'lithium',
    'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH measured', 'T3 measured', 'TT4 measured',
    'T4U measured', 'FTI measured', 'TBG measured'
    ]
    def to_binary(x):
        return 1 if x in ["t","True"] else 0
    vec1=[to_binary(doc1[col]) for col in binary_col]
    vec2=[to_binary(doc2[col]) for col in binary_col]
    f00=f01=f10=f11=0
    for i in range(len(vec1)):
        if vec1[i]==0 and vec2[i]==0:
            f00+=1
        elif vec1[i]==0 and vec2[i]==1:
            f01+=1
        elif vec1[i]==1 and vec2[i]==0:
            f10+=1
        else:
            f11+=1
    jc=f11/(f01+f10+f11)
    smc=(f11+f00)/(f00+f01+f10+f11)
    if smc>jc:
        return "JC is appropriate than SMC"
    return jc, smc

print("A6:")
# A6
def cosine_similarity():
    num_data=df1.select_dtypes(include=['int64','float64']) # taking only numeric data
    num_data=num_data.apply(pd.to_numeric,errors="coerce") # for removing missing values and changing to NAN
    num_data = num_data.fillna(0) # converting NAN to 0 because missing value has no contribution
    A=num_data.iloc[0].values
    B=num_data.iloc[1].values
    dot_AB=np.dot(A,B)
    norm_A=np.linalg.norm(A)
    norm_B=np.linalg.norm(B)
    return dot_AB/(norm_A*norm_B)

if __name__== "__main__":
    price = df["Price"]
    print(mean_wednesday(price))
    print(wed_price_mean_compare(price))
    print(apr_price_mean_compare(price))
    print(prob_loss(price))
    print(profit_on_wed(price))
    print(profit_if_today_wed(price))
    print(scatter_plot())
    print(mean_var_std())
    print(similarity_measure())
    print(cosine_similarity())

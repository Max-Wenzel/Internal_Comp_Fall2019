import pandas as pd

ttrac = pd.read_csv("final_test_data.csv")
ttrac = ttrac.drop(columns=["well_id", "treatment_condition"])
print(ttrac.head())
for i in range(len(ttrac)):
    if i % 2000 == 0:
        print(i)
    mean=ttrac.iloc[i].mean()
    std=ttrac.iloc[i].std()
    for j in range(600):
        ttrac.at[i,"ts_"+str(j + 1)]=((ttrac.at[i,"ts_" + str(j + 1)]-mean)/std)

ttrac.to_csv("normtest.csv", header=None)

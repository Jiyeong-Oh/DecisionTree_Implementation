import sys

train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

####################################################
#Settings
####################################################
import numpy as np
file_path = '{}'.format(train_file_name)
file_path2 = '{}'.format(test_file_name)

# reading train file
with open(file_path) as f:
    lines = f.read().splitlines()
train_db = [i.split('\t') for i in lines]
train_db = np.array([np.array(i) for i in train_db])

# reading test file
with open(file_path2) as f:
    lines = f.read().splitlines()
answer = lines
test_db = [i.split('\t') for i in lines]
test_db = np.array([np.array(i) for i in test_db])

# get ready to write the output file
f = open("./{}".format(output_file_name), 'w')


####################################################
# Building Decision Tree
####################################################
# every categories for each attribute
attr_unique = []
for i in range(len(train_db[0,:-1])):
     attr_unique.append(list(np.unique(train_db[1:,i])))

# checking whether the attribute has been selected
initial_check = np.zeros(len(attr_unique))


#Tree
def decision_tree(DB, attr_unique, used_check, potential_class=None):
    attr = len(attr_unique)
    
    def gini_index(DB):
        class_D = DB[1:,-1]
        gini = 1-sum([(list(class_D).count(i)/len(class_D))**2 for i in np.unique(class_D)])
        return gini

    def child_make(attr_num, j):
        attr_cat = attr_unique[attr_num] # [low, med, high]
        child = [np.array(DB[0])] 
        tmp = [np.array(x) for x in DB if x[attr_num]==attr_cat[j]]
        child.extend(tmp)
        child = np.array(child)
        return child

    def gini_attr(attr_num):
        attr_cat = attr_unique[attr_num] # [low, med, high]
        gini_child = 0
        for j in range(len(attr_cat)):
            child_DB = child_make(attr_num, j)
            dj_by_d = list(DB[1:,attr_num]).count(attr_cat[j])/len(DB[1:,attr_num])
            gini_dj = dj_by_d * gini_index(child_DB)
            gini_child += gini_dj
            
        return gini_child

    def reduction_in_impurity(DB, attr_num):
        gini_before = gini_index(DB)
        gini_after = gini_attr(attr_num)
        gini_gain = gini_before - gini_after
        return gini_gain

    def attr_select(DB, used_check):
        compare = [-10000 for i in range(attr)]
        for i in range(attr):
            if used_check[i] == 0:
                candidate = reduction_in_impurity(DB, i)
                compare[i] = candidate
        selected_attr = compare.index(max(compare))
        used_check[selected_attr] = 1
        return selected_attr
    
    def most_frequent(data):
        most_freq = max(data, key=data.count)
        most_freq_num = data.count(most_freq)
        for i in np.unique(data):
            if (i != most_freq) and (data.count(i)>=most_freq_num):
                return False
            return most_freq
    
    # 더 이상 나눌 수 없다: DB의 gini 계수가 0이거나, 남은 attr가 없거나, 아예 값이 없다
    if (len(np.unique(DB[1:,-1]))==1) :
        return DB[-1,-1]
    elif (0 not in used_check) or (len(DB)==1):
        return potential_class
    label = most_frequent(list(DB[1:,-1]))
    if label==False:
        label = potential_class
    
    used_check = used_check.copy()
    next_DB = attr_select(DB, used_check) # 선택된 attribute index
    child_list = []
    for k in range(len(attr_unique[next_DB])):
        child_list.append(child_make(next_DB, k))
        
    return {next_DB:{attr_unique[next_DB][i]: decision_tree(child_list[i], attr_unique, used_check, label) for i in range(len(attr_unique[next_DB]))}}

# tree is being made here
dt = decision_tree(train_db, attr_unique, initial_check)



####################################################
# Prediction
####################################################

def classify(tree, row):
    if (str(type(tree)) != "<class 'numpy.str_'>"):
        for key, value in tree.items():
            tree = value[row[key]]
            return classify(tree, row)
    return tree

answer_class = [] # predicted class
for row in test_db[1:]:
    answer_class.append(classify(dt,row))
    
####################################################
# Answer Writing
####################################################

answer[0] = answer[0]+'\t'+ train_db[0][-1]+'\n'
f.write("{}".format(answer[0]))
for i in range(len(answer)-1):
    answer[i+1] = answer[i+1]+'\t'+ answer_class[i]+'\n'
    f.write("{}".format(answer[i+1]))
f.close()

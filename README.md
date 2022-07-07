# DecisionTree_Implementation
## 결정나무 구현으로 카테고리 데이터 분류하기 (Classifying Categorical Data with Decision Tree)

여기서는 해당 이론을 바탕으로 Decision Tree를 파이썬으로 구현해 카테고리 데이터를 분류하는 과정을 포스팅하려고 한다.
- - -


구현은, 인자로 받은 train dataset과 test dataset 중 train dataset을 활용해 decision tree를 만들고, 만든 decision tree를 바탕으로 test dataset의 class를 예측하여 result dataset을 작성하는 알고리즘을 만드는 것이다.

작성한 알고리즘은 크게 아래 두 파트로 분류된다.





### (part 1) decision tree 생성

파트 1의 경우, attribute별로 불순도를 계산하여, reduction in impurity가 가장 큰 attribute를 기준으로 결정나무를 split하는 과정을 재귀적으로 반복하여 결정나무를 만드는 알고리즘을 수행한다.

여기서, Gain Ratio나 entropy를 사용했을 경우 발생할 수 있는 divide by zero error를 방지하기 위해, attribute를 선택하는 기준으로는 지니 계수가 사용되었다.



### (part 2) 데이터 예측

파트 2의 경우, 딕셔너리 형태의 decision tree의 value를 따라가며, test dataset의 class의 예측값을 저장하는 함수를 test dataset의 행 별로 반복 수행하는 알고리즘을 수행한다.

- - -

## Part 1) decision tree 생성
먼저 결정나무 생성을 위한 몇 가지 함수들을 정의했다.

 



### 지니 계수 계산 함수

```
def gini_index(DB):
        class_D = DB[1:,-1]
        gini = 1-sum([(list(class_D).count(i)/len(class_D))**2 for i in np.unique(class_D)])
        return gini
```

데이터셋을 인자로 넣으면, 그 class column에 속한 데이터를 세어 계산된 지니 계수를 반환해주는 함수다.

 



### 자식 노드 생성 함수

```
def child_make(attr_num, j):
        attr_cat = attr_unique[attr_num] # [low, med, high]
        child = [np.array(DB[0])] 
        tmp = [np.array(x) for x in DB if x[attr_num]==attr_cat[j]]
        child.extend(tmp)
        child = np.array(child)
        return child
```
        
인자로 받은 attribute의 인덱스 번호와, 그 attribute가 가지고 있는 category의 개수를 인자로 받고, 데이터셋을 attribute를 기준으로 j 개의 자식노드로 나눠주는 함수다.

 



### gini_a 계산 함수

```
def gini_attr(attr_num):
        attr_cat = attr_unique[attr_num] # [low, med, high]
        gini_child = 0
        for j in range(len(attr_cat)):
            child_DB = child_make(attr_num, j)
            dj_by_d = list(DB[1:,attr_num]).count(attr_cat[j])/len(DB[1:,attr_num])
            gini_dj = dj_by_d * gini_index(child_DB)
            gini_child += gini_dj
            
        return gini_child
```

인자로 받은 attribute를 기준으로 데이터셋을 분류했을 때 계산되는 지니계수를 반환해주는 함수다.

위에서 정의한 gini_index 함수를, 각 자식 노드에 적용해 계산하고, DB에서 각 카테고리가 차지하는 비율을 곱해 더하는 방식으로 gini_a를 구하는 것을 확인할 수 있다.





### reduction in impurity 계산 함수

```
def reduction_in_impurity(DB, attr_num):
        gini_before = gini_index(DB)
        gini_after = gini_attr(attr_num)
        gini_gain = gini_before - gini_after
        return gini_gain
```

위에서 계산한 gini와 gini_child의 차이를 구해주는 함수다.

 



### attribute 선택 함수
```
def attr_select(DB, used_check):
        compare = [-10000 for i in range(attr)]
        for i in range(attr):
            if used_check[i] == 0:
                candidate = reduction_in_impurity(DB, i)
                compare[i] = candidate
        selected_attr = compare.index(max(compare))
        used_check[selected_attr] = 1
        return selected_attr
```

각 attribute를 기준으로 데이터를 나눴을 때의 reduction in impurity를 계산하고, 각 값을 비교해 가장 불순도가 크게 감소한 attribute를 선택해주는 함수다.

이때, 이미 부모 노드에서 선택된 attribute가 다시 선택되지 않도록 확인하는 별도의 리스트를 함수 밖에 만들어 놓았다.

Attribute 선택 시 해당 리스트에 표기되도록 코드가 구성되어 있음을 확인할 수 있다.





### 다수 class 선택 함수

```
def most_frequent(data):
        most_freq = max(data, key=data.count)
        most_freq_num = data.count(most_freq)
        for i in np.unique(data):
            if (i != most_freq) and (data.count(i)>=most_freq_num):
                return False
            return
```

노드를 더 이상 나눌 수 없는 경우 중 부모 노드의 결과를 따라가야 하는 경우를 대비하여 만든 함수다.

데이터의 class 중 다수의 class가 무엇인지 계산하여 반환하도록 되어있다.

만약 데이터를 구성하고 있는 다수 class와 개수가 동일한 다른 class가 있다면, 함수 밖에서 처리하도록 false를 반환한다.



위에서 정의한 함수들은 모두 상위 함수 ‘decision tree’에 정의되어 있다.

정의한 함수들을 바탕으로, decision tree를 만드는 과정은 다음과 같이 설계되었다.

```
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
```

데이터셋을 더 이상 나눌 수 없는 경우를 정의하고, 각각의 경우 수행되어야 하는 알고리즘을 작성했다.



#### 1)    DB의 gini계수가 0인 경우

데이터셋의 class가 한 종류 뿐인 경우다. 이 때는 해당 클래스를 예측값으로 반환한다.

#### 2)    남은 attribute가 없는 경우

부모에서 attribute를 모두 소진한 경우다. 이 때는 다수의 클래스를 예측값으로 반환한다.

#### 3)    DB에 남은 데이터가 하나도 없는 경우

DB가 빈 데이터셋인 경우다. 이때는 부모노드의 결과를 따라간다.



지니계수를 바탕으로 Attribute를 선택하여 데이터를 나누고, 나눈 데이터셋에 대해 이전 과정을 재귀적으로 반복해주는 코드임을 확인할 수 있다.



반환값인 decision tree는 nested dictionary 형태다.

```
dt = decision_tree(train_db, attr_unique, initial_check)
```

모든 함수를 정의한 후, decision tree 함수를 실행해주었다.


## Part 2) 데이터 예측


만든 decision tree를 따라가며, test data 각 행의 클래스 라벨을 예측해주는 알고리즘이다.

```
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
```

먼저 하나의 행에 대해 tree를 따라가 마주한 leaf 노드 값을 반환해주는 함수 classify를 정의하고, for문을 돌면서 모든 행의 예측값들을 리스트에 저장해주었다.

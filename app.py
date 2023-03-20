
# Importing Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn import preprocessing 
from scipy import stats
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# label_encoder object knows how to understand word labels. 
    
#utility function
def split_data(X, y, test_size=0.1, seed=None):
    # shuffle data
    np.random.seed(seed)
    perm = np.random.permutation(X.index)
    X = X.loc[perm]
    y = y.loc[perm]
    
    # split into training and test sets
    n_samples = X.shape[0]
    if isinstance(test_size, float):
        if test_size <= 0 or test_size >= 1:
            raise ValueError("The test size should fall in the range (0,1)")
        n_train = n_samples - round(test_size*n_samples)
    elif isinstance(test_size, int):
        n_train = n_samples - test_size
    else:
        raise ValueError("Improper type \'%s\' for test_size" % type(test_size))

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test

def encode_one_hot(data): # note: pd.get_dummies(df) does the same
    # https://www.kite.com/python/answers/how-to-do-one-hot-encoding-with-numpy-in-python
    one_hot = np.zeros((data.size, data.max()+1))
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    return one_hot

def check_RandomState(random_state):
    """ Parse different input types for the random state"""
    if  random_state is None: 
        rng = np.random.RandomState() 
    elif isinstance(random_state, int): 
        # seed the random state with this integer
        rng = np.random.RandomState(random_state) 
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        raise ValueError ("improper type \'%s\' for random_state parameter" % type(random_state))
    return rng

def check_sample_size(sample_size, n_samples: int):
    if sample_size is None:
        n = n_samples
    elif isinstance(sample_size, int):
        if sample_size == 1:
            warnings.warn("Interpreting sample_size as 1 sample. Use sample_size=1.0 for 100% of the data")
        n = min(sample_size, n_samples)
    elif isinstance(sample_size, float):
        frac = min(sample_size, 1)
        n = int(frac*n_samples)
    else:
        raise ValueError("Improper type \'%s\' for sample_size" %type(sample_size))
    return n

def confusion_matrix(y_actual, y_pred):
    """ Returns a confusion matrix where the rows are the actual classes, and the columns are the predicted classes"""
    if y_actual.shape != y_pred.shape:
        raise ValueError ("input arrays must have the same shape, {}!={}".format(y_actual.shape, y_pred.shape))
    n = max(max(y_actual), max(y_pred)) + 1
    C = np.zeros((n, n), dtype=int)
    for label_actual in range(n):
        idxs_true = (y_actual == label_actual)
        for label_pred in range(n):
            C[label_actual, label_pred] = sum(y_pred[idxs_true] == label_pred)
    return C

def calc_f1_score(y_actual, y_pred) -> Tuple[float]:
    C = confusion_matrix(y_actual, y_pred)
    if C.shape[0] != 2:
        raise ValueError ("input arrays must only have binary values")
    recall    = C[1][1]/(C[1][0]+C[1][1]) #true positive/actual positive
    precision = C[1][1]/(C[0][1]+C[1][1]) #true positive/predicted positive
    if (recall == 0) or (precision == 0):
        f1 = 0
    else:
        f1 = 2 * recall*precision/(recall + precision) # = 2/((1/recall)+(1/precision))

    print("Recall: {:.4f}".format(recall))
    print("Precision: {:.4f}".format(precision))
    print("F1 Score: {:.4f}".format(f1))   


#Decision Tree function
import warnings
def gini_score(counts: List[int]) -> float: 
    score = 1
    n = sum(counts)
    for c in counts:
        p = c/n
        score -= p*p
    return score

class DecisionTree:
    def __init__(self, max_depth=None, max_features=None, min_samples_leaf=1, random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.RandomState = check_RandomState(random_state)

        # initialise internal variables
        self.tree_ = BinaryTree() 
        self.n_samples = []
        self.values = []
        self.impurities = []
        self.split_features = []
        self.split_values = []
        self.n_features = None
        self.n_classes = None
        self.features = None
        self.size = 0 # current node = size - 1
        
    def fit(self, X, Y):
        if Y.ndim == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable
        elif Y.shape[1] == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable

        # set internal variables
        self.n_features = X.shape[1]
        self.n_classes = Y.shape[1]
        self.features = X.columns
        self.max_depth_ = float('inf') if self.max_depth is None else self.max_depth
        if self.max_features is not None:
            if self.max_features is 'sqrt':
                n = int(np.sqrt(self.n_features))
            elif isinstance(self.max_features, int):
                n = min(self.max_features, self.n_features)
            else:
                raise Exception('Unknown parameter \'%s\' for max_features' % self.max_features)
        else:
            n = self.n_features 
        self.n_features_split = n

        # initial split which recursively calls itself
        self._split_node(X, Y) 

        # set attributes
        self.feature_importances_ = self.impurity_feature_importance()
        self.depths = self.tree_.find_depths()

    def get_n_splits(self) -> int:
        "The number of nodes (number of parameters/2) not counting the leaves in the tree"
        return self.tree_.n_splits
    
    def get_n_leaves(self) -> int:
        "The number of leaves (nodes without children) in the tree"
        return self.tree_.n_leaves

    def get_max_depth(self) -> int:
        "The maximum depth in the tree"
        return self.tree_.get_max_depth(0)

    def is_leaf(self, node_id: int) -> bool:
        return self.tree_.is_leaf(node_id)
    
    def split_name(self, node_id: int) -> str:
        return self.features[self.split_features[node_id]]

    def _set_defaults(self, node_id: int, Y):
        val = Y.sum(axis=0)
        self.values.append(val)
        self.impurities.append(gini_score(val))
        self.split_features.append(None)
        self.split_values.append(None)
        self.n_samples.append(Y.shape[0])
        self.tree_.add_node()
    
    def _split_node(self, X0, Y0):
        stack = [(X0, Y0, -1, -1, 0)] # stack = [(X, Y, parent, side, depth)]
        while stack:
            X, Y, parent, side, depth = stack.pop()
            node_id = self.size
            self.size += 1
            self._set_defaults(node_id, Y)
            if side is 0:
                self.tree_.set_left_child(parent, node_id)
            elif side is 1:
                self.tree_.set_right_child(parent, node_id)
            if self.impurities[node_id] == 0: # only one class in this node
                continue
            
            # random shuffling removes any bias due to the feature order
            features = self.RandomState.permutation(self.n_features)[:self.n_features_split]

            # make the split
            best_score = float('inf')
            for i in features:
                best_score= self._find_bettersplit(i, X, Y, node_id, best_score)
            if best_score == float('inf'):
                continue


            # make children
            if depth < self.max_depth_: 
                x_split = X.values[:, self.split_features[node_id]]
                lhs = np.nonzero(x_split<=self.split_values[node_id])
                rhs = np.nonzero(x_split> self.split_values[node_id])
                stack.append((X.iloc[rhs], Y[rhs[0], :], node_id, 1, depth+1)) # right first in, last out
                stack.append((X.iloc[lhs], Y[lhs[0], :], node_id, 0, depth+1)) # left first out

    
    def _find_bettersplit(self, var_idx: int, X, Y, node_id: int, best_score:float) -> float:
        X = X.values[:, var_idx] 
        n_samples = self.n_samples[node_id]

        # sort the variables. 
        order = np.argsort(X)
        X_sort, Y_sort = X[order], Y[order, :]

        #Start with all on the right. Then move one sample to left one at a time
        rhs_count = Y.sum(axis=0)
        lhs_count = np.zeros(rhs_count.shape)
        for i in range(0, len(X_sort)-1):
            xi, yi = X_sort[i], Y_sort[i, :]
            lhs_count += yi;  rhs_count -= yi
            if (xi == X_sort[i+1]) or (sum(lhs_count) < self.min_samples_leaf):
                continue
            if sum(rhs_count) < self.min_samples_leaf:
                break
            # Gini Impurity
            lhs_gini = gini_score(lhs_count)
            rhs_gini = gini_score(rhs_count)
            curr_score = (lhs_gini * lhs_count.sum() + rhs_gini * rhs_count.sum())/n_samples
            if curr_score < best_score:
                self.split_features[node_id] = var_idx
                best_score = curr_score
                self.split_values[node_id]= (xi + X_sort[i+1])/2
        return best_score

    def _predict_row(self, xi):
        next_node = 0
        while not self.is_leaf(next_node):
            left, right = self.tree_.get_children(next_node)
            next_node = left if xi[self.split_features[next_node]] <= self.split_values[next_node] else right
        return self.values[next_node]

    def _predict_batch(self, X, node=0):
        # Helper function for predict_prob(). Predicts multiple batches of a row at time. Faster than _predict_row(self, xi)
        if self.is_leaf(node):
            return self.values[node]
        if len(X) == 0:
            return np.empty((0, self.n_classes))
        left, right = self.tree_.get_children(node)

        lhs = X[:, self.split_features[node]] <= self.split_values[node]
        rhs = X[:, self.split_features[node]] >  self.split_values[node]

        probs = np.zeros((X.shape[0], self.n_classes))
        probs[lhs] = self._predict_batch(X[lhs], node=left)
        probs[rhs] = self._predict_batch(X[rhs], node=right)
        return probs

    def predict_prob(self, X):
        "Return the probability in the final leaf for each class, given as the fraction of each class in that leaf"
        if X.values.ndim == 1:
            probs = np.array([self._predict_row(X)])
        else:
            #start_time = time.time()
            #probs = np.apply_along_axis(self._predict_row, 1, X.values) # slow because this is a for loop
            probs = self._predict_batch(X.values)
            #end_time = time.time()
            #print('%.1fms' % ((end_time-start_time)*1000))
            probs /= np.sum(probs, axis=1)[:, None]
        return probs

    def predict(self, X):
        "Return the most likely class in the final leaf"
        probs = self.predict_prob(X)
        return np.nanargmax(probs, axis=1)

    def predict_count(self, X):
        "Return the sample count in the final leaf for each class"
        if X.values.ndim == 1:
            return np.array([self._predict_row(X.values)])
        return np.apply_along_axis(self._predict_row, 1, X.values)

    def impurity_feature_importance(self):
        """Calculate feature importance weighted by the number of samples affected by this feature at each split point. """
        feature_importances = np.zeros(self.n_features)
        total_samples = self.n_samples[0]
        for node in range(len(self.impurities)):
            if self.is_leaf(node):
                continue 
            spit_feature = self.split_features[node]
            impurity = self.impurities[node]
            n_samples = self.n_samples[node]
            # calculate score
            left, right = self.tree_.get_children(node)
            lhs_gini = self.impurities[left]
            rhs_gini = self.impurities[right]
            lhs_count = self.n_samples[left]
            rhs_count = self.n_samples[right]
            score = (lhs_gini * lhs_count + rhs_gini * rhs_count)/n_samples
            # feature_importances      = (decrease in node impurity) * (probability of reaching node ~ proportion of samples)
            feature_importances[spit_feature] += (impurity-score) * (n_samples/total_samples)

        feature_importances = feature_importances/feature_importances.sum()
        return feature_importances

    def get_info(self, node_id: int):
        n_samples =  self.n_samples[node_id]
        val =        self.values[node_id]
        impurity =   self.impurities[node_id]
        var_idx    = self.split_features[node_id]
        split_val  = self.split_values[node_id]
        if self.is_leaf(node_id):
            return n_samples, val, impurity
        else:
            return n_samples, val, var_idx, split_val, impurity

    def node_to_string(self, node_id: int) -> str:
        if self.is_leaf(node_id):
            n_samples, val, impurity = self.get_info(node_id)
            s = 'n_samples: {:d}; value: {}; impurity: {:.4f}'.format(n_samples, val.astype(int), impurity)
        else:
            n_samples, val, var_idx, split_val, impurity = self.get_info(node_id)
            split_name = self.split_name(node_id)
            s =  'n_samples: {:d}; value: {}; impurity: {:.4f}'.format(n_samples, val.astype(int), impurity)
            s += '; split: {}<={:.3f}'.format(split_name, split_val)
        return s
  

class BinaryTree():
    def __init__(self):
        self.children_left = []
        self.children_right = []

    @property
    def size(self):
        "The number of nodes in the tree"
        return len(self.children_left)

    @property
    def n_leaves(self):
        "The number of leaves (nodes without children) in the tree"
        return self.children_left.count(-1) 

    @property
    def n_splits(self):
        "The number of nodes (number of parameters/2) not counting the leaves in the tree"
        return self.size - self.n_leaves

    def add_node(self):
        self.children_left.append(-1)
        self.children_right.append(-1)
    
    def set_left_child(self, node_id: int, child_id: int):
        self.children_left[node_id] = child_id

    def set_right_child(self, node_id: int, child_id: int):
        self.children_right[node_id] = child_id

    def get_children(self, node_id: int): return self.children_left[node_id], self.children_right[node_id]

    def find_depths(self):
        depths = np.zeros(self.size, dtype=int)
        depths[0] = -1
        stack = [(0, 0)] # (parent, node_id)
        while stack:
            parent, node_id = stack.pop()
            if node_id == -1:
                continue
            depths[node_id] = depths[parent] + 1
            left = self.children_left[node_id]
            right = self.children_right[node_id]
            stack.extend([(node_id, left), (node_id, right)])
        return depths

    def is_leaf(self, node_id: int):
        left = self.children_left[node_id]
        right = self.children_right[node_id]
        return right == left #(left == -1) and (right == -1)

    def get_max_depth(self, node_id=0):
        "Calculate the maximum depth of the tree"
        if self.is_leaf(node_id):
            return 0 
        left = self.children_left[node_id]
        right = self.children_right[node_id]
        return max(self.get_max_depth(left), self.get_max_depth(right)) + 1

    def preorder(self, node_id=0):
        "Pre-order tree traversal"
        # Note: the parallel arrays are already in pre-order
        # Therefore can just return np.arange(self.size)
        if node_id != -1:
            yield node_id
        left = self.children_left[node_id]
        right = self.children_right[node_id]
        if left != -1:
            for leaf in self.preorder(left):
                yield leaf
        if right != -1:
            for leaf in self.preorder(right):
                yield leaf

class RandomForestClassifier:
    def __init__(self, 
                n_trees=100, 
                random_state=None, 
                max_depth=None, 
                max_features=None, 
                min_samples_leaf=1,
                sample_size=None, 
                bootstrap=True, 
                oob_score=False):
        self.n_trees = n_trees
        self.RandomState = check_RandomState(random_state)
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf=min_samples_leaf
        self.sample_size = sample_size
        self.bootstrap = bootstrap
        self.oob_score = oob_score

        self.features = None
        self.n_features = None
        self.n_classes = None
        self.feature_importances_ = None
        
    def fit(self, X, Y):
        "fit the random tree to the independent variable X, to determine the dependent variable Y"
        if Y.ndim == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable
        elif Y.shape[1] == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable

        # set internal variables
        self.n_features = X.shape[1]
        self.n_classes = Y.shape[1]
        self.features = X.columns.values
        n_samples = X.shape[0]
        self.sample_size_ = check_sample_size(self.sample_size, n_samples)

        # create decision trees
        self.trees = []
        rng_states = [] # save the random states to regenerate the random indices for the oob_score
        for i in range(self.n_trees):
            rng_states.append(self.RandomState.get_state())
            self.trees.append(self._create_tree(X, Y))

        # set attributes
        self.feature_importances_ = self.impurity_feature_importances()
        if self.oob_score:
            if not (self.bootstrap or (self.sample_size_<n_samples)):
                warnings.warn("out-of-bag score will not be calculated because bootstrap=False")
            else:
                self.oob_score_ = self.calculate_oob_score(X, Y, rng_states)
    
    def _create_tree(self, X, Y):
        assert len(X) == len(Y), ""
        n_samples = X.shape[0]

        # get sub-sample 
        if self.bootstrap:
            rand_idxs = self.RandomState.randint(0, n_samples, self.sample_size_) # with replacement
            X_, Y_ = X.iloc[rand_idxs, :], Y[rand_idxs] # approximate unique values =n*(1-np.exp(-sample_size_/n_samples))
        elif self.sample_size_ < n_samples:
            rand_idxs = self.RandomState.permutation(np.arange(n_samples))[:self.sample_size_]  # without replacement
            X_, Y_ = X.iloc[rand_idxs, :], Y[rand_idxs]
        else:
            X_, Y_ = X.copy(), Y.copy() # do nothing to the data

        new_tree =  DecisionTree(max_depth=self.max_depth, 
                                 max_features=self.max_features,
                                 random_state=self.RandomState,
                                 min_samples_leaf=self.min_samples_leaf
                                )
        new_tree.fit(X_, Y_)
        return new_tree
                
    def predict(self, X):
        "Predict the class for each sample in X"
        probs = np.sum([t.predict_prob(X) for t in self.trees], axis=0)
        #probs = np.sum([t.predict_count(X) for t in self.trees], axis=0)
        return np.nanargmax(probs, axis=1)

    def score(self, X, y):
        "The accuracy score of random forest predictions for X to the true classes y"
        y_pred = self.predict(X)
        return np.mean(y_pred==y)

    def calculate_oob_score(self, X, Y, rng_states):
        n_samples = X.shape[0]
        oob_prob = np.zeros(Y.shape)
        oob_count = np.zeros(n_samples)
        all_samples = np.arange(n_samples)
        rng = np.random.RandomState()
        for i, state in enumerate(rng_states):
            rng.set_state(state)
            if self.bootstrap:
                rand_idxs = rng.randint(0, n_samples, self.sample_size_)
            else: #self.sample_size_ < n_samples
                rand_idxs = rng.permutation(all_samples)[:self.sample_size_]
            row_oob = np.setxor1d(all_samples, rand_idxs)
            oob_prob[row_oob, :] += self.trees[i].predict_prob(X.iloc[row_oob])
            oob_count[row_oob] += 1
        # remove nan-values
        valid = oob_count > 0 
        oob_prob = oob_prob[valid, :]
        oob_count = oob_count[valid][:, np.newaxis] # transform to column vector for broadcasting during the division
        y_test    =  np.argmax(Y[valid], axis=1)
        # predict out-of-bag score
        y_pred = np.argmax(oob_prob/oob_count, axis=1)
        return np.mean(y_pred==y_test)

    def impurity_feature_importances(self) -> np.ndarray:
        """Calculate feature importance weighted by the number of samples affected by this feature at each split point. """
        feature_importances = np.zeros((self.n_trees, self.n_features))

        for i, tree in enumerate(self.trees):
            feature_importances[i, :] = tree.feature_importances_

        return np.mean(feature_importances, axis=0)
    

        

#Label encoding
label_encoder = preprocessing.LabelEncoder() 
data=''
# Setting up the header 
st.title("Dataset")
st.subheader("Complete Model Lifecycle")


Choose_file  = st.selectbox("Select filfe upload type", ("Single_file", "Two_file",))

if Choose_file== "Single_file":
    filename = st.file_uploader("upload file", type = ("csv", "xlsx"))
    data = pd.read_csv(filename,na_values=['?', '/', '#',''])
elif Choose_file =='Two_file':

    # Upload the first dataset
    df1 = st.file_uploader("Upload the first dataset", type=["csv", "xlsx"])

    # Upload the second dataset
    df2 = st.file_uploader("Upload the second dataset", type=["csv", "xlsx"])

    # Merge the two datasets
    if df1 is not None and df2 is not None:
        df1 = pd.read_csv(df1,na_values=['?', '/', '#','']) # Use pd.read_excel(df1) for Excel files
        df2 = pd.read_csv(df2,na_values=['?', '/', '#','']) # Use pd.read_excel(df2) for Excel files
        data = pd.merge(df1, df2, on='ID')
        st.write(data)
    else:
        st.write("Pleasse upload both datasets.")


# Providing a radio button to browse and upload the imput file 


#------------------------------------------------------------------------------
# To upload an input file from the specified path
#@st.cache(persist=True)
#def explore_data(dataset):
#    df = pd.read_csv(os.path.join(dataset))
#    return df
#data = explore_data(my_dataset)
#------------------------------------------------------------------------------
def mean_squared_error1(y_true, y_pred):
   
      # Check if the lengths of both arrays are equal
      if len(y_true) != len(y_pred):
          raise ValueError("Length of y_true and y_pred should be the same.")
      
      # Calculate the squared differences between the true and predicted values
      squared_differences = [(y_true[i] - y_pred[i])**2 for i in range(len(y_true))]
      
      # Calculate the mean of the squared differences
      mse1 = sum(squared_differences) / len(squared_differences)
      print(mse1)
      return mse1

from sklearn.metrics import r2_score

def r2(y_true, y_pred):
    # Calculate the mean of the true values
    y_true_mean = sum(y_true) / len(y_true)

    # Calculate the total sum of squares (TSS)
    tss = sum((y_true - y_true_mean) ** 2)

    # Calculate the residual sum of squares (RSS)
    rss = sum((y_true - y_pred) ** 2)

    # Calculate the R-squared value
    r2_score = 1-(rss / tss)

    return r2_score

def remove_outliers(data):
    z_scores = np.abs(stats.zscore(data))
    data_clean = data[(z_scores < 3).all(axis=1)]
    return data_clean


def fill_outliers(data, method='zscore', axis=0):
    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        threshold = 3
        data[z_scores > threshold] = np.nan
        data.fillna(data.median(), inplace=True)
    elif method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        data[(data < lower_bound) | (data > upper_bound)] = np.nan
        data.fillna(data.median(), inplace=True)

    if axis == 0:
        return data
    elif axis == 1:
        return data.T

def drop_outliers(data, method='zscore', axis=0):
    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        threshold = 3
        data[z_scores > threshold] = np.nan
        data.dropna(axis=axis, inplace=True)
    elif method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        data[(data < lower_bound) | (data > upper_bound)] = np.nan
        data.dropna(axis=axis, inplace=True)

    return data

# Dataset preview
if st.checkbox("Preview Dataset"):
    if st.button("Head"):
        st.write(data.head())
    elif st.button("Tail"):
        st.write(data.tail())
    else:
        number = st.slider("Select No of Rows", 1, data.shape[0])
        st.write(data.head(number))


# show entire data
if st.checkbox("Show all data"):
    st.write(data)

st.subheader('To Check Columns Name')
# show column names
if st.checkbox("Show Column Names"):
    st.write(data.columns)

# show dimensions
if st.checkbox("Show Dimensions"):
    st.write(data.shape)

st.subheader('Summary of the Data')     
# show summary
if st.checkbox("Show Summary"):
    st.write(data.describe())

numeric_columns = data.select_dtypes(include=['int', 'float'])
st.subheader('Check null values and fill null values ')   
# show missing values
if st.checkbox("Show Missing Values"):
    st.write(numeric_columns.isna().sum())    

# Select a column to treat missing values
col_option = st.multiselect("Select Feature to fillna",numeric_columns.columns)

# Specify options to treat missing values
missing_values_clear = st.selectbox("Select Missing values treatment method", ("Replace with Mean", "Replace with Median", "Replace with Mode"))

if missing_values_clear == "Replace with Mean":
    replaced_value = data[col_option].mean()
    data[col_option]=data[col_option].mean()
    st.write("Mean value of column is :", replaced_value)
elif missing_values_clear == "Replace with Median":
    replaced_value = data[col_option].median()
    st.write("Median value of column is :", replaced_value)
elif missing_values_clear == "Replace with Mode":
    replaced_value = data[col_option].mode()
    st.write("Mode value of column is :", replaced_value)


Replace = st.selectbox("Replace values of column?", ("Yes", "No"))
if Replace == "Yes":
    data[col_option] = data[col_option].fillna(replaced_value,)
    st.write("Null values replaced")
elif Replace == "No":
    st.write("No changes made")

st.subheader(' Check Null values Categorical Columns and fill Null values  ')
#only categorical columns
object_columns = data.select_dtypes(include=['object'])
if st.checkbox("Show Missing Values of object columns"):
    st.write(object_columns.isna().sum()) 

col_option1 = st.multiselect("Select Feature to fillna",object_columns.columns)


# Specify options to treat missing values
missing_values_clear = st.selectbox("Select Missing values For Categorycal columns treatment method", ("Replace with Mean", "Replace with Median", "Replace with Mode"))

if missing_values_clear == "Replace with Mean":
    replaced_value1 = data[col_option1].mean()
    
    st.write("Mean value of column is :", replaced_value1)
elif missing_values_clear == "Replace with Median":
    replaced_value1 = data[col_option1].median()
    st.write("Median value of column is :", replaced_value1)
elif missing_values_clear == "Replace with Mode":
    replaced_value1 ='Missinig'
    
    st.write("Mode value of column is :", replaced_value1)



Replace = st.selectbox("Replace values of column to category?", ("Yes", "No"))
if Replace == "Yes":
    data[col_option1] = data[col_option1].fillna(replaced_value1,)
    st.write("Null values replaced")
elif Replace == "No":
    st.write("No changes made")



if st.checkbox("Show Missing   Values after fill"):
    st.write(data.isna().sum()) 
# To change datatype of a column in a dataframe
# display datatypes of all columns
if st.checkbox("Show datatypes of the columns"):
    st.write(data.dtypes)

st.subheader('Convert Datatype')
col_option_datatype = st.multiselect("Select Column to change datatype", data.columns) 

input_data_type = st.selectbox("Select Datatype of input column", (str,int, float))  
output_data_type = st.selectbox("Select Datatype of output column", (label_encoder,'OneHot_encode'))

st.write("Datatype of ",col_option_datatype," changed to ", output_data_type)
if output_data_type=='OneHot_encode':
    for i in col_option_datatype:
        data = pd.get_dummies(data, columns=[i],drop_first=True)
        
else:
    for i in col_option_datatype:
        data[i] = output_data_type.fit_transform(data[i])
        


if st.checkbox("Show updated datatypes of the columns"):
    st.write(data.dtypes)

if st.checkbox("Preview Dataset aftre convert datatype"):
    if st.button("Head "):
        st.write(data.head())

st.subheader(' Check Outliers and Replace Outliers')
show_outliers = st.checkbox("Show outliers")

# Display data with or without outliers
if show_outliers:
    for k, v in data.items():
            q1 = v.quantile(0.25)
            q3 = v.quantile(0.75)
            irq = q3 - q1
            v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
            perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
            print("Column %s outliers = %.2f%%" % (k, perc))
            st.write(k,perc)


method = st.selectbox("Select outlier detection method", ("IQR", "Z-score"))

if st.checkbox("Fill Outliers"):
    if method == "IQR":
        data = fill_outliers(data, method='iqr', axis=0)
    elif method == "Z-score":
        data = fill_outliers(data, method='zscore', axis=0)

    st.write("Data with filled outliers")
    st.write(data)

if st.checkbox("Drop Outliers"):
    if method == "IQR":
        data = drop_outliers(data, method='iqr', axis=0)
    elif method == "Z-score":
        data = drop_outliers(data, method='zscore', axis=0)

    st.write("Data with dropped outliers")
    st.write(data)




show_outliers = st.checkbox("Show outliers aftre treatement")

# Display data with or without outliers
if show_outliers:
    for k, v in data.items():
            q1 = v.quantile(0.25)
            q3 = v.quantile(0.75)
            irq = q3 - q1
            v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
            perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
            print("Column %s outliers = %.2f%%" % (k, perc))
            st.write(k,perc)
# visualization
st.subheader('Scatter Plot')
# scatter plot
col1 = st.selectbox('Which feature on x?', data.columns)
col2 = st.selectbox('Which feature on y?', data.columns)
fig = px.scatter(data, x =col1,y=col2)
st.plotly_chart(fig)

st.subheader('Correlation Plot') 
# correlartion plots
if st.checkbox("Show Correlation plots with Seaborn"):
    st.write(sns.heatmap(data.corr()))
    st.pyplot()

st.subheader('Feature_Scaling')
scaling_method = st.selectbox('Select a scaling method:', ['Standardization', 'Normalization'])

# Perform the selected scaling method on the dataset
if scaling_method == 'Standardization':
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
else:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

# Display the scaled data
st.write('Scaled data:')
st.write(pd.DataFrame(scaled_data, columns=data.columns))

# Machine Learning Algorithms
st.subheader('Machine Learning models')
 
from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
 
 
features = st.multiselect("Select Feature Columns",data.columns)
labels = st.multiselect("select target column",data.columns)

features= data[features].values
labels = data[labels].values


train_percent = st.slider("Select % to train model", 1, 100)
train_percent = train_percent/100

X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=train_percent, random_state=1)



alg = ['Smote','Random Forest Classifier']
classifier = st.selectbox('Which algorithm?', alg)
   
if classifier == 'Smote':
    sm = SMOTEENN()
    X_resampled, y_resampled = sm.fit_resample(features,labels)
    xr_train,xr_test,yr_train,yr_test=train_test_split(X_resampled, y_resampled,test_size=0.3)
    model_dt_smote=RandomForestClassifier()
    model_dt_smote.fit(xr_train,yr_train)
    yr_pred = model_dt_smote.predict(xr_test)
    acc = model_dt_smote.score(xr_test, yr_pred)
    st.write('Accuracy: ', acc)
    cm=confusion_matrix(yr_test,yr_pred)
    st.write('Confusion matrix: ', cm)

elif classifier == 'Random Forest Classifier':
    RFC=RandomForestClassifier()
    RFC.fit(X_train, y_train)
    acc = RFC.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_RFC = RFC.predict(X_test)
    cm=confusion_matrix(y_test,pred_RFC)
    st.write('Confusion matrix: ', cm)




# html_temp = """
#     <body style="background-image: url("F:\Dockers-master\g1.jpg");
#     background-size: cover;">
#     <div style="background-color:tomato;padding:10px">
#     <h2 style="color:white;text-align:center;">Streamlit Student Performance Analysis ML App </h2>
#     </div>
#     </body>
#     """
    
# st.markdown(html_temp,unsafe_allow_html=True)
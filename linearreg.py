import numpy as np 

class LinearRegression:
    
    def __init__(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of rows in X and y must be equal")
        
        if type(X) != np.ndarray:
            raise ValueError("X must be a numpy array")
        if type(y) != np.ndarray:
            raise ValueError("y must be a numpy array")
        
        self.X = X
        self.y = y
    
    
    def check_low_rank(self):
        if self.X.shape[0] < self.X.shape[1]:
            # raise ValueError("Number of rows in X must be greater than the number of columns")
            return True
        return False
    
    def check_full_rank(self):
        if np.linalg.matrix_rank(self.X) == self.X.shape[1]:
            return True
        return False
    
    def fit(self):
        self.prepare_data()
        
        if not self.check_low_rank() and self.check_full_rank():
            self.w = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
        else:
            print("X is not full rank")
            
        
    def prepare_data(self):
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)
        
        self.X = (self.X-self.mean)/self.std
        self.X = np.column_stack((np.ones(self.X.shape[0]), self.X))


        
    def predict(self, X_test):
        if type(X_test) != np.ndarray:
            raise ValueError("X_test must be a numpy array") 
        X_test = (X_test-self.mean)/self.std
        X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))
        return X_test.dot(self.w)
    
    
    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true-y_pred)**2))
from carla import MLModel

class TFModelAdapter:
    '''Adapter to plug TFModel into CFEC explainers'''
    def __init__(self, model: MLModel, backend: str) -> None:
        '''
        Parameters:

            `model`: Trained MLModel  
            `backend`: either 'tensorflow' or 'sklearn' 
        '''
        self.model = model

        if backend  in ['tensorflow', 'sklearn']:   
            self.backend = backend
        else:
            raise

    
    def __call__(self, data):
        return self.predict(data)

    def predict(self, data):
        if self.backend == 'tensorflow':
            return self.model.predict(data)
        elif self.backend == 'sklearn':
            return self.model.predict_proba(data)

    

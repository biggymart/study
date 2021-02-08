# %% [markdown]
# ### Now we use edafa (TTA package)
# Step 1: Import the predictor suitable for your problem (`ClassPredictor` for Classification and `SegPredictor` for Segmentation)
from edafa import ClassPredictor

# %% [markdown]
# Step 2: Inherit predictor class and implement the main function `predict_patches(self,patches)`

class myPredictor(ClassPredictor):
    def __init__(self,model,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = model

    def predict_patches(self,patches):
        return self.model.predict(patches)

# %% [markdown]
# Step 3: Instantiate your class with configuration and whatever parameters needed
# use orignal image and flipped Left-Right images
# use arithmetic mean for averaging
conf = '{"augs":["NO", "FLIP_LR"], "mean":"ARITH"}'


p = myPredictor(model,conf)

# %% [markdown]
# Step 4: Predict images
y_pred_aug = p.predict_images(X_val)
y_pred_aug = [(y[0]>=0.5).astype(np.uint8) for y in y_pred_aug ]

print('Accuracy with TTA:',np.mean((y_val==y_pred_aug)))

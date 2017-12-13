# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:08:10 2017

@author: Stefan
"""

(X_train, y_train, c) = datafile.get_batch(9)

fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio
ax = [fig.add_subplot(3,3,i+1) for i in range(9)]

for i, a in enumerate(ax):
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.set_aspect('equal')
    a.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    a.axis('off')

fig.subplots_adjust(wspace=-0.02, hspace=0.02)
fig.show()
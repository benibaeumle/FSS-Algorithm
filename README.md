# A Fast Shapelet Selection Algorithm for Time Series Classification
A python numpy implementation of the shapelet candidate selection algorithm from the paper Ji et al., „A Shapelet Selection Algorithm for Time Series Classification“.
See the [paper](https://www.sciencedirect.com/science/article/abs/pii/S1389128618312970) for a more detailed description.
# How to use

```python
shapelet_selection = ShapeletCandidateSelection(n_lfdp = n_lfdp)
shapelet_candidates = shapelet_selection.transform(data)
```
See the [demo](https://github.com/benibaeumle/A-Shapelet-Selection-Algorithm/blob/main/demo/demo.ipynb) for a more 
detailed example.

# License
Released under MIT License. The code was implemented by Benedikt Bäumle.

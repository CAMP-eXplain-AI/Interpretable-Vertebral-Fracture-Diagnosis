from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import numpy as np

def get_confusion_matrix_figure(y_true, y_pred, labels=None, title="Confusion matrix"):
    """
    Returns a confusion matrix plot.
    """

    plt.figure(dpi=600)
    label_codes = np.arange(len(labels)) if labels is not None else None
    cm = confusion_matrix(y_true, y_pred, labels=label_codes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    values_format = None # Format specification for values in confusion matrix. If `None`, the format specification is 'd' or '.2g' whichever is shorter.
    disp.plot(
        include_values=True,
        cmap=plt.cm.Blues, # 'viridis'
        ax=None, 
        xticks_rotation='horizontal',
        values_format=values_format
    )
    fig = disp.figure_
    fig.suptitle(title)
    return fig
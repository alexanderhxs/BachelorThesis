
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
boxes = [
    {
        'label' : "Male height",
        'whislo': 162.6,    # Bottom whisker position
        'q1'    : 170.2,    # First quartile (25th percentile)
        'med'   : 175.7,    # Median         (50th percentile)
        'q3'    : 180.4,    # Third quartile (75th percentile)
        'whishi': 187.8,    # Top whisker position
        'fliers': [100, 200]        # Outliers
    }
]
ax.bxp(boxes, showfliers=False)
ax.set_ylabel("cm")
plt.show()
import matplotlib.pyplot as plt

# Mock feature importance data
features = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research']
importances = [0.15, 0.10, 0.08, 0.12, 0.10, 0.30, 0.15]

# Create bar chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(features, importances, color='steelblue')
ax.set_xlabel("Importance")
ax.set_title("Feature Importance Chart")
plt.tight_layout()

# Save the image
fig.savefig("feature_importance_plot.png")
print("Saved as 'feature_importance_plot.png'")

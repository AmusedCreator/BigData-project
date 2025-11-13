from level_remote_correlation_plot import level_remote_correlation_plot
from salary_by_direction_plot import salary_by_direction_plot
from specialization_skills_analysis_plot import specialization_skills_analysis_plot

def analyze_all():
    level_remote_correlation_plot()
    salary_by_direction_plot()
    specialization_skills_analysis_plot()

if __name__ == '__main__':
    analyze_all()
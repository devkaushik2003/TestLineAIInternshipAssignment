import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import requests  # Added this import at the top of the file
import os
import urllib.request
import urllib.error
import ssl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import seaborn as sns

class QuizAnalyzer:
    def __init__(self):
        self.historical_data = []
        self.current_quiz = None
        
    def load_data(self, historical_data: List[Dict], current_quiz: Dict):
        """Load quiz data into the analyzer"""
        self.historical_data = historical_data
        self.current_quiz = current_quiz
        
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze overall performance metrics"""
        performance = {
            'total_quizzes': len(self.historical_data),
            'average_accuracy': self._calculate_average_accuracy(),
            'topic_performance': self._analyze_topic_performance(),
            'strength_areas': [],
            'weak_areas': [],
            'improvement_trends': self._analyze_improvement_trends()
        }
        return performance
    
    def _calculate_average_accuracy(self) -> float:
        """Calculate average accuracy across all quizzes"""
        accuracies = [
            float(quiz['accuracy'].replace(' %', ''))  # Changed from strip to replace
            for quiz in self.historical_data
        ]
        return sum(accuracies) / len(accuracies) if accuracies else 0
    
    def _analyze_topic_performance(self) -> Dict[str, Dict]:
        """Analyze performance by topic"""
        topic_stats = {}
        
        for quiz in self.historical_data:
            topic = quiz['quiz']['topic']
            accuracy = float(quiz['accuracy'].replace(' %', ''))  # Ensure accuracy is float
            score = float(quiz['score']) if isinstance(quiz['score'], str) else quiz['score']  # Convert score if it's string
            
            if topic not in topic_stats:
                topic_stats[topic] = {
                    'attempts': 0,
                    'total_accuracy': 0,
                    'scores': []
                }
            
            topic_stats[topic]['attempts'] += 1
            topic_stats[topic]['total_accuracy'] += accuracy
            topic_stats[topic]['scores'].append(score)
            
        # Calculate averages and identify trends
        for topic in topic_stats:
            stats = topic_stats[topic]
            stats['average_accuracy'] = stats['total_accuracy'] / stats['attempts']
            stats['average_score'] = sum(stats['scores']) / len(stats['scores'])
            
        return topic_stats
    
    def _analyze_improvement_trends(self) -> Dict[str, Any]:
        """Analyze improvement trends over time"""
        sorted_quizzes = sorted(
            self.historical_data,
            key=lambda x: datetime.strptime(x['submitted_at'], '%Y-%m-%dT%H:%M:%S.%f%z')
        )
        
        trends = {
            'accuracy_trend': [],
            'score_trend': [],
            'speed_trend': []
        }
        
        for quiz in sorted_quizzes:
            trends['accuracy_trend'].append(float(quiz['accuracy'].strip(' %')))
            trends['score_trend'].append(quiz['score'])
            trends['speed_trend'].append(float(quiz['speed']))
            
        return trends
    
    def generate_recommendations(self) -> List[str]:
        """Generate personalized recommendations based on analysis"""
        performance = self.analyze_performance()
        recommendations = []
        
        # Analyze overall performance
        if performance['average_accuracy'] < 70:
            recommendations.append(
                "Focus on improving overall accuracy through careful question reading "
                "and concept revision"
            )
        
        # Topic-specific recommendations
        topic_stats = performance['topic_performance']
        for topic, stats in topic_stats.items():
            if stats['average_accuracy'] < 60:
                recommendations.append(
                    f"Dedicate more study time to {topic} - current accuracy is below target"
                )
            elif stats['average_accuracy'] > 90:
                recommendations.append(
                    f"Maintain strong performance in {topic} through periodic revision"
                )
        
        # Time management recommendations
        if any(float(quiz['speed']) < 90 for quiz in self.historical_data):  # Convert speed to float
            recommendations.append(
                "Work on improving time management during quizzes to attempt more questions"
            )
        
        return recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        performance = self.analyze_performance()
        recommendations = self.generate_recommendations()
        
        return {
            'performance_summary': performance,
            'recommendations': recommendations,
            'student_persona': self._determine_student_persona(performance)
        }
    
    def _determine_student_persona(self, performance: Dict) -> str:
        """Determine student persona based on performance patterns"""
        avg_accuracy = performance['average_accuracy']
        
        if avg_accuracy >= 90:
            return "Advanced Learner"
        elif avg_accuracy >= 70:
            return "Steady Performer"
        elif avg_accuracy >= 50:
            return "Growing Learner"
        else:
            return "Needs Additional Support"

    def save_report_to_files(self, report: Dict[str, Any], output_dir: str = "reports"):
        """Save the analysis report to separate files"""
        # Create reports directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save performance summary
        performance_file = os.path.join(output_dir, f"performance_summary_{timestamp}.json")
        with open(performance_file, 'w') as f:
            json.dump(report['performance_summary'], f, indent=4)
        
        # Save recommendations
        recommendations_file = os.path.join(output_dir, f"recommendations_{timestamp}.txt")
        with open(recommendations_file, 'w') as f:
            f.write("Performance Report\n")
            f.write("=================\n")
            f.write(f"Student Persona: {report['student_persona']}\n\n")
            f.write("Recommendations:\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        return performance_file, recommendations_file

    def visualize_performance(self, output_dir: str = "reports/visualizations") -> None:
        """Generate and save performance visualizations as static images"""
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use a simple style instead of seaborn
        plt.style.use('default')
        
        # 1. Topic Performance Bar Chart
        plt.figure(figsize=(12, 6))
        topic_stats = self._analyze_topic_performance()
        topics = list(topic_stats.keys())
        accuracies = [stats['average_accuracy'] for stats in topic_stats.values()]
        
        bars = plt.bar(topics, accuracies, color='skyblue')
        plt.title('Performance by Topic', fontsize=14, pad=20)
        plt.xlabel('Topics', fontsize=12)
        plt.ylabel('Average Accuracy (%)', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'topic_performance_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Improvement Trends Line Chart
        plt.figure(figsize=(12, 6))
        trends = self._analyze_improvement_trends()
        
        plt.plot(trends['accuracy_trend'], marker='o', label='Accuracy', linewidth=2, color='blue')
        plt.plot(trends['score_trend'], marker='s', label='Score', linewidth=2, color='green')
        
        plt.title('Performance Trends Over Time', fontsize=14, pad=20)
        plt.xlabel('Quiz Number', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'improvement_trends_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Speed vs Accuracy Scatter Plot
        plt.figure(figsize=(10, 8))
        speeds = [float(quiz['speed']) for quiz in self.historical_data]
        accuracies = [float(quiz['accuracy'].replace(' %', '')) for quiz in self.historical_data]
        
        plt.scatter(speeds, accuracies, alpha=0.5, color='purple')
        
        # Add trend line
        z = np.polyfit(speeds, accuracies, 1)
        p = np.poly1d(z)
        plt.plot(speeds, p(speeds), "r--", alpha=0.8)
        
        plt.title('Speed vs Accuracy Analysis', fontsize=14, pad=20)
        plt.xlabel('Speed', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'speed_accuracy_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Student Profile Radar Chart
        performance = self.analyze_performance()
        categories = ['Accuracy', 'Speed', 'Consistency', 'Topic Coverage', 'Improvement']
        
        # Calculate metrics
        values = [
            performance['average_accuracy'],
            sum(float(quiz['speed']) for quiz in self.historical_data) / len(self.historical_data),
            100 - np.std([float(quiz['accuracy'].replace(' %', '')) for quiz in self.historical_data]),
            len(performance['topic_performance']) * 20,  # Normalize to 100
            (trends['accuracy_trend'][-1] - trends['accuracy_trend'][0]) + 50  # Normalize around 50
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))  # Repeat the first value to close the polygon
        angles = np.concatenate((angles, [angles[0]]))  # Repeat the first angle to close the polygon
        categories = np.concatenate((categories, [categories[0]]))
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.set_ylim(0, 100)
        
        plt.title('Student Performance Profile', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'student_profile_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Strength/Weakness Analysis
        plt.figure(figsize=(12, 6))
        topic_accuracies = [(topic, stats['average_accuracy']) for topic, stats in topic_stats.items()]
        topics, accuracies = zip(*sorted(topic_accuracies, key=lambda x: x[1], reverse=True))
        
        colors = ['green' if acc >= 70 else 'red' for acc in accuracies]
        plt.bar(topics, accuracies, color=colors, alpha=0.6)
        plt.axhline(y=70, color='black', linestyle='--', alpha=0.5, label='Proficiency Line')
        
        plt.title('Strength and Weakness Analysis', fontsize=14, pad=20)
        plt.xlabel('Topics', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'strength_weakness_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Create SSL context that ignores certificate verification
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    analyzer = QuizAnalyzer()
    
    try:
        # Fetch data using urllib
        with urllib.request.urlopen("https://api.jsonserve.com/XgAgFJ", context=ctx) as response:
            historical_data = json.loads(response.read())
        
        with urllib.request.urlopen("https://jsonkeeper.com/b/LLQT", context=ctx) as response:
            current_quiz = json.loads(response.read())
        
        analyzer.load_data(historical_data, current_quiz)
        report = analyzer.generate_report()
        
        # Save report to files
        performance_file, recommendations_file = analyzer.save_report_to_files(report)
        print(f"Performance summary saved to: {performance_file}")
        print(f"Recommendations saved to: {recommendations_file}")
        
        # Generate visualizations
        analyzer.visualize_performance()
        
        print("Visualizations have been saved to the reports directory")
        
    except urllib.error.URLError as e:
        print(f"Error fetching data: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
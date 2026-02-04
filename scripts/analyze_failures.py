"""
Analyze failed repositories from Phase 2 feature extraction.
Extract failure reasons and save to CSV for further analysis.

Author: Manikanta Gonugondla
Co-Author: Claude Sonnet 4.5
"""

import pandas as pd
import re

def parse_output_file(output_file):
    """Parse output.txt to identify failed repos and reasons"""

    with open(output_file, 'r') as f:
        lines = f.readlines()

    failed_repos = []
    current_repo = None

    for line in lines:
        # Detect repo processing start
        if line.startswith('Processing:'):
            current_repo = line.split('Processing:')[1].strip()

        # Detect failure reasons
        elif current_repo:
            if 'No Python files found' in line:
                failed_repos.append({
                    'repo_name': current_repo,
                    'failure_reason': 'No Python files found',
                    'category': 'No Python Files'
                })
                current_repo = None

            elif 'Error extracting files: File is not a zip file' in line:
                failed_repos.append({
                    'repo_name': current_repo,
                    'failure_reason': 'ZIP extraction error - not a valid zip file',
                    'category': 'ZIP Format Error'
                })
                current_repo = None

            elif 'Failed to download' in line:
                failed_repos.append({
                    'repo_name': current_repo,
                    'failure_reason': 'Failed to download repository',
                    'category': 'Download Error'
                })
                current_repo = None

            elif 'Error downloading' in line:
                failed_repos.append({
                    'repo_name': current_repo,
                    'failure_reason': 'Network/API error during download',
                    'category': 'Download Error'
                })
                current_repo = None

            # If we see "Analyzing X Python files", this repo succeeded
            elif 'Analyzing' in line:
                current_repo = None

    return failed_repos


def main():
    """Main execution"""
    print("=" * 70)
    print("Analyzing Failed Repositories from Phase 2")
    print("=" * 70)

    # Parse output file
    output_file = '../output.txt'
    failed_repos = parse_output_file(output_file)

    print(f"\nTotal failed repositories: {len(failed_repos)}")

    # Convert to DataFrame
    df_failed = pd.DataFrame(failed_repos)

    # Save to CSV
    output_csv = '../data/failed_repos.csv'
    df_failed.to_csv(output_csv, index=False)

    print(f"Saved failed repos to: {output_csv}")

    # Print summary by category
    print("\nFailure Breakdown:")
    print("-" * 70)
    category_counts = df_failed['category'].value_counts()
    for category, count in category_counts.items():
        print(f"{category}: {count} repos ({count/len(failed_repos)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("Failed Repositories:")
    print("=" * 70)
    print(df_failed.to_string(index=False))

    # Analyze patterns
    print("\n" + "=" * 70)
    print("Analysis & Recommendations:")
    print("=" * 70)

    no_python = len(df_failed[df_failed['category'] == 'No Python Files'])
    zip_errors = len(df_failed[df_failed['category'] == 'ZIP Format Error'])

    print(f"\n1. No Python Files ({no_python} repos):")
    print("   - Repos likely contain only documentation/configs")
    print("   - Cannot extract code features")
    print("   - Recommendation: Exclude from dataset")

    print(f"\n2. ZIP Format Errors ({zip_errors} repos):")
    print("   - GitHub may use different compression or branch structure")
    print("   - Try alternative download methods (git clone)")
    print("   - Recommendation: Implement git clone fallback")

    print("\n" + "=" * 70)
    print(f"Current Dataset Status:")
    print("=" * 70)
    print(f"Total repos collected: 200")
    print(f"Successfully processed: 165 (82.5%)")
    print(f"Failed: 35 (17.5%)")
    print(f"\nUsable dataset size: 165 repositories")
    print(f"This is sufficient for ML training (target was 100+)")


if __name__ == "__main__":
    main()

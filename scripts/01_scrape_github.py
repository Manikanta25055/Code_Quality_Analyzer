"""
Phase 1: GitHub Repository Data Collection
Collects high-quality and low-quality Python repositories from GitHub.

Author: Manikanta Gonugondla
Co-Author: Claude Sonnet 4.5
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

# GitHub API configuration
GITHUB_API_URL = "https://api.github.com/search/repositories"
GITHUB_TOKEN = None  # Add your GitHub token here for higher rate limits (keep it private!)

class GitHubRepoCollector:
    def __init__(self, token=None):
        self.token = token
        self.headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        if token:
            self.headers['Authorization'] = f'token {token}'

    def search_repos(self, query, max_repos=200):
        """Search GitHub repositories with given query"""
        repos = []
        per_page = 30
        max_pages = (max_repos // per_page) + 1

        for page in range(1, max_pages + 1):
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': per_page,
                'page': page
            }

            try:
                response = requests.get(GITHUB_API_URL, headers=self.headers, params=params)

                # Check rate limit
                if response.status_code == 403:
                    print(f"Rate limit exceeded. Waiting 60 seconds...")
                    time.sleep(60)
                    continue

                response.raise_for_status()
                data = response.json()

                if 'items' not in data:
                    break

                repos.extend(data['items'])
                print(f"Collected {len(repos)} repositories so far...")

                if len(repos) >= max_repos:
                    break

                # Be nice to GitHub API
                time.sleep(2)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching repos: {e}")
                break

        return repos[:max_repos]

    def collect_high_quality_repos(self, count=100):
        """Collect high-quality repositories
        Criteria: 500+ stars, recent commits, has tests
        """
        print(f"\nCollecting {count} high-quality repositories...")

        # Recent date (within last 3 months)
        recent_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

        query = f"language:python stars:>500 pushed:>{recent_date}"
        repos = self.search_repos(query, count)

        return self.process_repos(repos, quality_label=1)

    def collect_low_quality_repos(self, count=100):
        """Collect low-quality repositories
        Criteria: <50 stars, no recent commits (1+ year)
        """
        print(f"\nCollecting {count} low-quality repositories...")

        # Old date (no commits in last year)
        old_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        query = f"language:python stars:<50 pushed:<{old_date}"
        repos = self.search_repos(query, count)

        return self.process_repos(repos, quality_label=0)

    def process_repos(self, repos, quality_label):
        """Process repository data and create structured format"""
        processed = []

        for repo in repos:
            repo_data = {
                'repo_name': repo['full_name'],
                'repo_url': repo['html_url'],
                'clone_url': repo['clone_url'],
                'stars': repo['stargazers_count'],
                'forks': repo['forks_count'],
                'last_pushed': repo['pushed_at'],
                'description': repo['description'],
                'quality_label': quality_label  # 1 = high quality, 0 = low quality
            }
            processed.append(repo_data)

        return processed

    def save_to_csv(self, high_quality_repos, low_quality_repos, output_file):
        """Save collected repositories to CSV"""
        all_repos = high_quality_repos + low_quality_repos
        df = pd.DataFrame(all_repos)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(all_repos)} repositories to {output_file}")

        # Print summary
        print(f"\nDataset Summary:")
        print(f"High-quality repos (label=1): {len(high_quality_repos)}")
        print(f"Low-quality repos (label=0): {len(low_quality_repos)}")
        print(f"Total: {len(all_repos)}")

        return df


def main():
    """Main execution function"""
    print("=" * 70)
    print("GitHub Repository Data Collection - Phase 1")
    print("=" * 70)

    # Initialize collector
    collector = GitHubRepoCollector(token=GITHUB_TOKEN)

    print("\nStarting full collection: 200 repositories (100 high + 100 low)")

    high_quality = collector.collect_high_quality_repos(count=100)
    low_quality = collector.collect_low_quality_repos(count=100)

    # Save to CSV
    output_file = '../data/labels.csv'
    df = collector.save_to_csv(high_quality, low_quality, output_file)

    # Display sample
    print("\nSample of collected data:")
    print(df[['repo_name', 'stars', 'quality_label']].head(10))

    print("\n" + "=" * 70)
    print("Phase 1 Complete - Data Collection Successful!")
    print("=" * 70)
    print(f"\nNext Steps:")
    print("1. Verify data/labels.csv has 200 repositories")
    print("2. Review dataset balance (100 high-quality, 100 low-quality)")
    print("3. Move to Phase 2: Feature Extraction")


if __name__ == "__main__":
    main()

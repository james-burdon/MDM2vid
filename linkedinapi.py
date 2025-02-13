#objective extract files from this to get more data with company descriptions and start sorting them a bit 

"""
Example script demonstrating how to search for jobs on LinkedIn
"""

import json
import time
from linkedin_api import Linkedin
import os
import sys
import re

TEST_LINKEDIN_USERNAME = os.environ.get("LINKEDIN_USERNAME")  # Correct
TEST_LINKEDIN_PASSWORD = os.environ.get("LINKEDIN_PASSWORD")  # Correct
print(f"Username: {TEST_LINKEDIN_USERNAME}")
print(f"Password: {TEST_LINKEDIN_PASSWORD}")


def main():
    if not (TEST_LINKEDIN_USERNAME and TEST_LINKEDIN_PASSWORD):
        print("Test config incomplete. Exiting...")
        sys.exit()

    # Initialize the API client
    api = Linkedin(TEST_LINKEDIN_USERNAME, TEST_LINKEDIN_PASSWORD, refresh_cookies=True)

    # Example search parameters
    search_params = {
        "keywords": "Python Developer",
        "location_name": "San Francisco Bay Area",
        "remote": ["2"],  # Remote jobs only
        "experience": ["2", "3"],  # Entry level and Associate
        "job_type": ["F", "C"],  # Full-time and Contract
        "limit": 5,
    }

    try:
        # Perform the job search
        jobs = api.search_jobs(**search_params)
        # Process and display results
        print(f"\nFound {len(jobs)} jobs matching your criteria:")

        for job in jobs:
            try:
                job_id = job["entityUrn"].split(":")[-1]
                details = api.get_job(job_id)

                title = details.get("title", "Unknown")
                # Print full job details for debugging
                print(json.dumps(details, indent=2))

                # Extract company name from multiple possible sources
                company = (
                    details.get("companyDetails", {}).get("name") or
                    details.get("hiringCompany", {}).get("name") or
                    details.get("organization", {}).get("name") or
                    details.get("employer", {}).get("name") or
                    details.get("company", {}).get("name") or
                    details.get("companyName") or
                    "Unknown"
                )

                # Extract company ID in case name is missing
                company_id = details.get("companyDetails", {}).get("company", {}).get("id")

                if company == "Unknown" and company_id:
                    try:
                        company_info = api.get_company(company_id)  # Fetch company details
                        company = company_info.get("name", "Unknown Company")
                    except Exception as e:
                        print(f"Could not fetch company details: {e}")

                print(f"Extracted Company: {company}")  # Debugging print statement
                location = details.get("formattedLocation", "Unknown")
                remote = details.get("workRemoteAllowed", "Unknown")

                # Extract description
                description_data = details.get("description", {})
                if "text" in description_data:
                    description = description_data["text"]
                else:
                    description = " ".join(
                        attr.get("text", "") for attr in description_data.get("attributes", []) if "text" in attr
                    )

                if not description:
                    description = "No description available"

                # Clean description formatting
                description = re.sub(r'\s+', ' ', description).strip()
                description = re.sub(r'\{.*?pemberly.*?\}', '', description)
                description = re.sub(r'\{.*?attributeKindUnion.*?\}', '', description)

                # Extract skills from job details
                skills = details.get("jobDetails", {}).get("skills", [])

                # Try fetching skills from the API if missing
                if not skills:
                    try:
                        skills_data = api.get_job_skills(job_id)
                        skills = [
                            skill.get("skill", {}).get("name", "Unknown") 
                            for skill in skills_data.get("skillMatchStatuses", [])
                        ]
                    except Exception as e:
                        print("Could not fetch skills:", str(e))
                        skills = []

                print("\n-------------------")
                print(f"Title: {title}")
                print(f"Company: {company}")
                print(f"Location: {location}")
                print(f"Remote? {remote}")
                print(f"Description: {description}")

                print("\nRequired Skills:")
                if skills:
                    for skill in skills:
                        print(f"- {skill}")
                else:
                    print("No skills listed.")

            except Exception as e:
                print(f"Error processing job {job_id}: {str(e)}")

    except Exception as e:
        print(f"Error performing job search: {str(e)}")


def cache_job_data(job_id, job_data):
    """Cache job data to a JSON file"""
    try:
        # Create a cache directory if it doesn't exist
        import os

        os.makedirs("cache/jobs", exist_ok=True)

        cache_file = f"cache/jobs/job_{job_id}.json"
        with open(cache_file, "w") as f:
            json.dump(job_data, f, indent=2)
        print(f"Job data cached to {cache_file}")

    except Exception as e:
        print(f"Error caching job data: {str(e)}")


def get_workplace_type_string(type_code):
    """Convert workplace type code to readable string"""
    types = {1: "On-site", 2: "Remote", 3: "Hybrid"}
    return types.get(type_code, "Unknown")


if __name__ == "__main__":
    main()

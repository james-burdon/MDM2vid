"""
Example script demonstrating how to search for jobs on LinkedIn
"""

import json
import time
from linkedin_api import Linkedin
import os
import sys

TEST_LINKEDIN_USERNAME = os.environ.get("LINKEDIN_USERNAME")
TEST_LINKEDIN_PASSWORD = os.environ.get("LINKEDIN_PASSWORD")


def main():
    if not (TEST_LINKEDIN_USERNAME and TEST_LINKEDIN_PASSWORD):
        print("Test config incomplete. Exiting...")
        sys.exit()

    api = Linkedin(TEST_LINKEDIN_USERNAME, TEST_LINKEDIN_PASSWORD)
    jobs = api.search_jobs(keywords="Python Developer", location_name="San Francisco Bay Area", limit=5)
    
    for job in jobs:
        job_id = job["entityUrn"].split(":")[-1]
        details = api.get_job(job_id)
        
        # Extract company name from multiple possible sources
        # Extract company name with deeper lookup
        company = (
            details.get("companyResolutionResult", {}).get("name") or
            details.get("companyDetails", {})
            .get("com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany", {})
            .get("companyResolutionResult", {})
            .get("name") or
            "Unknown"
        )


        # Extract job description
        description_data = details.get("description", {})
        
        if "text" in description_data:
            description = description_data["text"]
        else:
            description = " ".join(
                attr.get("text", "").strip()
                for attr in description_data.get("attributes", [])
                if "text" in attr
            )
        
        description = " ".join(description.split())
        
        print("\n-------------------")
        print(f"Title: {details.get('title', 'unknown')}")
        print(f"Company: {company}")
        print(f"Location: {details.get('formattedLocation', 'unknown')}")
        print(f"Description: {description if description else 'No description available'}")
        
        time.sleep(2)

if __name__ == "__main__":
    main()


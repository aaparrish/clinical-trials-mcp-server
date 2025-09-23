#!/usr/bin/env python3
"""
Clinical Trials MCP Server

This MCP server provides tools to search and retrieve clinical trial information
from ClinicalTrials.gov API v2.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime, UTC

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server instance
server = Server("clinical-trials")

# Constants
CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
MAX_RESULTS_PER_REQUEST = 100


def safe_get(d: dict, *keys) -> Any:
    """Safely get nested dictionary values."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur


def simplify_locations(locations_raw: List[Dict]) -> List[str]:
    """Extract location strings from ClinicalTrials.gov location objects."""
    if not locations_raw:
        return []
    
    locations = []
    for loc in locations_raw:
        if not isinstance(loc, dict):
            continue
        
        parts = []
        
        # Get facility name
        facility = loc.get("facility")
        if isinstance(facility, str):
            parts.append(facility)
        elif isinstance(facility, dict):
            name = facility.get("name")
            if name:
                parts.append(name)
        
        # Get city
        city = loc.get("city")
        if city:
            parts.append(city)
        
        # Get state
        state = loc.get("state")
        if state:
            parts.append(state)
        
        # Get country (only if not US)
        country = loc.get("country")
        if country and country != "United States":
            parts.append(country)
        
        if parts:
            locations.append(", ".join(parts))
    
    return locations


def extract_trial_info(study: Dict) -> Dict[str, Any]:
    """Extract key information from a clinical trial study."""
    protocol = study.get("protocolSection", {})
    
    # Basic identification
    identification = protocol.get("identificationModule", {})
    nct_id = identification.get("nctId", "Unknown")
    brief_title = identification.get("briefTitle", "No title available")
    official_title = identification.get("officialTitle", "")
    
    # Status information
    status_module = protocol.get("statusModule", {})
    overall_status = status_module.get("overallStatus", "Unknown")
    start_date_struct = status_module.get("startDateStruct", {})
    start_date = start_date_struct.get("date", "Unknown") if start_date_struct else "Unknown"
    
    completion_date_struct = status_module.get("primaryCompletionDateStruct", {})
    completion_date = completion_date_struct.get("date", "Unknown") if completion_date_struct else "Unknown"
    
    # Study design
    design_module = protocol.get("designModule", {})
    study_type = design_module.get("studyType", "Unknown")
    phases = design_module.get("phases", [])
    phase_str = ", ".join(phases) if phases else "Not specified"
    
    # Conditions
    conditions_module = protocol.get("conditionsModule", {})
    conditions = conditions_module.get("conditions", [])
    
    # Eligibility
    eligibility = protocol.get("eligibilityModule", {})
    gender = eligibility.get("gender", "Unknown")
    min_age = eligibility.get("minimumAge", "Unknown")
    max_age = eligibility.get("maximumAge", "Unknown")
    criteria = eligibility.get("eligibilityCriteria", "Not provided")
    
    # Locations
    contacts_locations = protocol.get("contactsLocationsModule", {})
    locations = contacts_locations.get("locations", [])
    simplified_locations = simplify_locations(locations)
    
    # Sponsors
    sponsors_module = protocol.get("sponsorCollaboratorsModule", {})
    lead_sponsor_info = sponsors_module.get("leadSponsor", {})
    lead_sponsor = lead_sponsor_info.get("name", "Unknown") if isinstance(lead_sponsor_info, dict) else str(lead_sponsor_info)
    
    # Interventions
    arms_interventions = protocol.get("armsInterventionsModule", {})
    interventions = arms_interventions.get("interventions", [])
    intervention_names = []
    for interv in interventions:
        if isinstance(interv, dict):
            name = interv.get("name", "Unknown")
            intervention_type = interv.get("type", "")
            if intervention_type:
                intervention_names.append(f"{name} ({intervention_type})")
            else:
                intervention_names.append(name)
    
    # Description
    description_module = protocol.get("descriptionModule", {})
    brief_summary = description_module.get("briefSummary", "No summary available")
    detailed_description = description_module.get("detailedDescription", "")
    
    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": official_title,
        "overall_status": overall_status,
        "study_type": study_type,
        "phase": phase_str,
        "conditions": conditions,
        "start_date": start_date,
        "completion_date": completion_date,
        "lead_sponsor": lead_sponsor,
        "gender": gender,
        "min_age": min_age,
        "max_age": max_age,
        "locations": simplified_locations,
        "interventions": intervention_names,
        "brief_summary": brief_summary,
        "detailed_description": detailed_description,
        "eligibility_criteria": criteria[:500] + "..." if len(criteria) > 500 else criteria,
        "url": f"https://clinicaltrials.gov/study/{nct_id}"
    }


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_clinical_trials",
            description="Search for clinical trials on ClinicalTrials.gov",
            inputSchema={
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "Medical condition or disease to search for (e.g., 'lung cancer', 'diabetes')",
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional location filter (city, state, or country)",
                        "default": None,
                    },
                    "status": {
                        "type": "string",
                        "description": "Trial recruitment status",
                        "enum": ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "TERMINATED", "SUSPENDED", "WITHDRAWN"],
                        "default": "RECRUITING",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of trials to return (1-100)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                    },
                    "intervention": {
                        "type": "string",
                        "description": "Optional intervention/treatment filter",
                        "default": None,
                    },
                },
                "required": ["condition"],
            },
        ),
        Tool(
            name="get_trial_details",
            description="Get detailed information about a specific clinical trial by NCT ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "nct_id": {
                        "type": "string",
                        "description": "NCT ID of the trial (e.g., 'NCT04267848')",
                    },
                },
                "required": ["nct_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    
    if name == "search_clinical_trials":
        return await search_clinical_trials(**arguments)
    elif name == "get_trial_details":
        return await get_trial_details(**arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def search_clinical_trials(
    condition: str,
    location: Optional[str] = None,
    status: str = "RECRUITING",
    max_results: int = 20,
    intervention: Optional[str] = None,
) -> List[TextContent]:
    """Search for clinical trials."""
    
    try:
        # Build query parameters
        params = {
            "query.cond": condition,
            "filter.overallStatus": status.upper(),
            "pageSize": min(max_results, MAX_RESULTS_PER_REQUEST),
            "format": "json"
        }
        
        if location:
            params["query.locn"] = location
        
        if intervention:
            params["query.intr"] = intervention
        
        # Make API request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(CTGOV_BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        # Process studies
        studies = data.get("studies", [])
        if not studies:
            return [TextContent(
                type="text",
                text=f"No trials found for condition: {condition}"
            )]
        
        # Extract trial information
        trials = []
        for study in studies:
            try:
                trial_info = extract_trial_info(study)
                trials.append(trial_info)
            except Exception as e:
                logger.warning(f"Error processing study: {e}")
                continue
        
        # Format response
        if not trials:
            return [TextContent(
                type="text",
                text=f"Found {len(studies)} studies but couldn't process any successfully"
            )]
        
        # Create summary
        summary = f"Found {len(trials)} clinical trials for '{condition}'"
        if location:
            summary += f" in {location}"
        summary += f" (Status: {status})\n\n"
        
        for i, trial in enumerate(trials, 1):
            summary += f"{i}. **{trial['brief_title']}**\n"
            summary += f"   - NCT ID: {trial['nct_id']}\n"
            summary += f"   - Status: {trial['overall_status']}\n"
            summary += f"   - Phase: {trial['phase']}\n"
            summary += f"   - Study Type: {trial['study_type']}\n"
            summary += f"   - Sponsor: {trial['lead_sponsor']}\n"
            
            if trial['interventions']:
                summary += f"   - Interventions: {', '.join(trial['interventions'][:2])}\n"
            
            if trial['locations']:
                locations_str = ', '.join(trial['locations'][:3])
                if len(trial['locations']) > 3:
                    locations_str += f" (+ {len(trial['locations'])-3} more)"
                summary += f"   - Locations: {locations_str}\n"
            
            summary += f"   - Eligibility: {trial['gender']}, Ages {trial['min_age']} to {trial['max_age']}\n"
            summary += f"   - URL: {trial['url']}\n"
            
            if trial['brief_summary'] and trial['brief_summary'] != "No summary available":
                brief = trial['brief_summary'][:200] + "..." if len(trial['brief_summary']) > 200 else trial['brief_summary']
                summary += f"   - Summary: {brief}\n"
            
            summary += "\n"
        
        return [TextContent(type="text", text=summary)]
        
    except httpx.HTTPError as e:
        logger.error(f"HTTP error searching trials: {e}")
        return [TextContent(
            type="text",
            text=f"Error searching clinical trials: {str(e)}"
        )]
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return [TextContent(
            type="text",
            text=f"Unexpected error occurred: {str(e)}"
        )]


async def get_trial_details(nct_id: str) -> List[TextContent]:
    """Get detailed information about a specific trial."""
    
    try:
        # Clean NCT ID
        if not nct_id.upper().startswith("NCT"):
            nct_id = f"NCT{nct_id}"
        
        # Make API request for specific study
        url = f"{CTGOV_BASE_URL}/{nct_id}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        
        # Extract detailed information
        study = data
        trial_info = extract_trial_info(study)
        
        # Format detailed response
        details = f"**Clinical Trial Details: {trial_info['nct_id']}**\n\n"
        details += f"**Title:** {trial_info['brief_title']}\n\n"
        
        if trial_info['official_title'] and trial_info['official_title'] != trial_info['brief_title']:
            details += f"**Official Title:** {trial_info['official_title']}\n\n"
        
        details += f"**Status:** {trial_info['overall_status']}\n"
        details += f"**Study Type:** {trial_info['study_type']}\n"
        details += f"**Phase:** {trial_info['phase']}\n"
        details += f"**Conditions:** {', '.join(trial_info['conditions']) if trial_info['conditions'] else 'Not specified'}\n"
        details += f"**Start Date:** {trial_info['start_date']}\n"
        details += f"**Completion Date:** {trial_info['completion_date']}\n"
        details += f"**Lead Sponsor:** {trial_info['lead_sponsor']}\n\n"
        
        if trial_info['interventions']:
            details += f"**Interventions:**\n"
            for intervention in trial_info['interventions']:
                details += f"- {intervention}\n"
            details += "\n"
        
        details += f"**Eligibility:**\n"
        details += f"- Gender: {trial_info['gender']}\n"
        details += f"- Age Range: {trial_info['min_age']} to {trial_info['max_age']}\n\n"
        
        if trial_info['locations']:
            details += f"**Locations ({len(trial_info['locations'])}):**\n"
            for location in trial_info['locations'][:10]:  # Show first 10
                details += f"- {location}\n"
            if len(trial_info['locations']) > 10:
                details += f"- ... and {len(trial_info['locations'])-10} more locations\n"
            details += "\n"
        
        if trial_info['brief_summary'] and trial_info['brief_summary'] != "No summary available":
            details += f"**Summary:**\n{trial_info['brief_summary']}\n\n"
        
        if trial_info['eligibility_criteria'] and trial_info['eligibility_criteria'] != "Not provided":
            details += f"**Eligibility Criteria:**\n{trial_info['eligibility_criteria']}\n\n"
        
        details += f"**More Information:** {trial_info['url']}\n"
        
        return [TextContent(type="text", text=details)]
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return [TextContent(
                type="text",
                text=f"Clinical trial {nct_id} not found"
            )]
        else:
            logger.error(f"HTTP error getting trial details: {e}")
            return [TextContent(
                type="text",
                text=f"Error retrieving trial {nct_id}: {str(e)}"
            )]
    except Exception as e:
        logger.error(f"Unexpected error getting trial details: {e}")
        return [TextContent(
            type="text",
            text=f"Unexpected error occurred: {str(e)}"
        )]


async def main():
    """Run the MCP server."""
    async with stdio_server() as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
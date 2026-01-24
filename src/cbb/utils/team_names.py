"""
Centralized team name mapping utilities.

This module provides consistent team name normalization and mapping between
different data sources (raw game data, KenPom, odds APIs, etc.).
"""

from __future__ import annotations

import re
from typing import Optional


# Comprehensive manual mappings for known problem cases
MANUAL_MAPPINGS: dict[str, str] = {
    # Alabama variations
    "Alabama Crimson Tide": "Alabama",
    "Alabama State Hornets": "Alabama St.",
    # Arkansas variations
    "Arkansas-Little Rock Trojans": "Little Rock",
    "Arkansas-Pine Bluff Golden Lions": "Arkansas Pine Bluff",
    # B
    "Bethune-Cookman Wildcats": "Bethune Cookman",
    # C
    "Cal State-Fullerton Titans": "Cal St. Fullerton",
    "Cal State-Northridge Matadors": "CSUN",
    "Central Connecticut State Blue Devils": "Central Connecticut",
    "Central Florida Knights": "UCF",
    # E
    "East Texas A&M Lions": "East Texas A&M",
    # F
    "FIU Panthers": "FIU",
    "Florida International Golden Panthers": "FIU",
    "Florida Atlantic Owls": "Florida Atlantic",
    # H
    "Hawaii Rainbow Warriors": "Hawaii",
    "Hawai'i Rainbow Warriors": "Hawaii",
    # I
    "Illinois-Chicago Flames": "Illinois Chicago",
    "UIC Flames": "Illinois Chicago",
    "IUPUI Jaguars": "IU Indy",
    "Indiana-Purdue-Indianapolis Jaguars": "IU Indy",
    # L
    "LIU Sharks": "LIU",
    "Long Island University Sharks": "LIU",
    "Louisiana Ragin Cajuns": "Louisiana",
    "Louisiana-Lafayette Ragin Cajuns": "Louisiana",
    "Louisiana-Monroe Warhawks": "Louisiana Monroe",
    "Loyola (IL) Ramblers": "Loyola Chicago",
    "Loyola Chicago Ramblers": "Loyola Chicago",
    "Loyola (MD) Greyhounds": "Loyola MD",
    "Loyola Marymount Lions": "Loyola Marymount",
    # M
    "Miami (FL) Hurricanes": "Miami FL",
    "Miami (OH) RedHawks": "Miami OH",
    "Mississippi Rebels": "Mississippi",
    "Ole Miss Rebels": "Mississippi",
    "Mississippi State Bulldogs": "Mississippi St.",
    # N
    "N.C. State Wolfpack": "N.C. State",
    "NC State Wolfpack": "N.C. State",
    "Nebraska-Omaha Mavericks": "Nebraska Omaha",
    "NJIT Highlanders": "NJIT",
    "North Carolina A&T Aggies": "North Carolina A&T",
    "North Carolina Central Eagles": "North Carolina Central",
    "North Carolina Tar Heels": "North Carolina",
    "Northern Kentucky Norse": "Northern Kentucky",
    # P
    "Penn Quakers": "Penn",
    "Pittsburgh Panthers": "Pittsburgh",
    "Purdue Fort Wayne Mastodons": "Purdue Fort Wayne",
    # S
    "Saint Francis Red Flash": "Saint Francis",
    "Saint Joseph's Hawks": "Saint Joseph's",
    "Saint Mary's Gaels": "Saint Mary's",
    "Saint Peter's Peacocks": "Saint Peter's",
    "Sam Houston Bearkats": "Sam Houston St.",
    "Sam Houston State Bearkats": "Sam Houston St.",
    "San Jose State Spartans": "San Jose St.",
    "SIU-Edwardsville Cougars": "SIUE",
    "SMU Mustangs": "SMU",
    "South Carolina State Bulldogs": "South Carolina St.",
    "South Florida Bulls": "South Florida",
    "Southeast Missouri State Redhawks": "Southeast Missouri",
    "Southeastern Louisiana Lions": "Southeastern Louisiana",
    "Southern Illinois Salukis": "Southern Illinois",
    "Southern Miss Golden Eagles": "Southern Miss",
    "Southern Mississippi Golden Eagles": "Southern Miss",
    "Stephen F. Austin Lumberjacks": "Stephen F. Austin",
    "St. Bonaventure Bonnies": "St. Bonaventure",
    "St. John's Red Storm": "St. John's",
    "St. Thomas Tommies": "St. Thomas",
    # T
    "TCU Horned Frogs": "TCU",
    "Texas A&M Aggies": "Texas A&M",
    "Texas A&M-Corpus Christi Islanders": "Texas A&M Corpus Chris",
    "Texas-Arlington Mavericks": "UT Arlington",
    "Texas-Rio Grande Valley Vaqueros": "UT Rio Grande Valley",
    "Texas-San Antonio Roadrunners": "UTSA",
    "Toledo Rockets": "Toledo",
    "Troy Trojans": "Troy",
    "Tulane Green Wave": "Tulane",
    "Tulsa Golden Hurricane": "Tulsa",
    # U
    "UAB Blazers": "UAB",
    "UC-Davis Aggies": "UC Davis",
    "UC-Irvine Anteaters": "UC Irvine",
    "UC-Riverside Highlanders": "UC Riverside",
    "UC-San Diego Tritons": "UC San Diego",
    "UC-Santa Barbara Gauchos": "UC Santa Barbara",
    "UCF Knights": "UCF",
    "UCLA Bruins": "UCLA",
    "UMass Minutemen": "Massachusetts",
    "UMass-Lowell River Hawks": "UMass Lowell",
    "UMBC Retrievers": "UMBC",
    "UNC-Asheville Bulldogs": "UNC Asheville",
    "UNC-Greensboro Spartans": "UNC Greensboro",
    "UNC-Wilmington Seahawks": "UNC Wilmington",
    "UNLV Rebels": "UNLV",
    "USC Trojans": "USC",
    "USC-Upstate Spartans": "USC Upstate",
    "UT-Martin Skyhawks": "Tennessee Martin",
    "UTEP Miners": "UTEP",
    # V
    "VCU Rams": "VCU",
    "VMI Keydets": "VMI",
    # W
    "William & Mary Tribe": "William & Mary",
}


def normalize_team_name(name: str) -> str:
    """
    Normalize a team name for consistent comparison.

    Args:
        name: Raw team name

    Returns:
        Normalized lowercase name with standardized spacing
    """
    # Strip and lowercase
    name = name.strip().lower()

    # Standardize common variations
    name = name.replace("st.", "state")
    name = name.replace("&", "and")
    name = re.sub(r"\s+", " ", name)  # Collapse multiple spaces

    return name


class TeamNameMapper:
    """
    Maps team names between different data sources.

    Usage:
        mapper = TeamNameMapper(kenpom_teams)
        kp_name = mapper.get_kenpom_name("Alabama Crimson Tide")
    """

    def __init__(
        self,
        kenpom_teams: set[str],
        additional_mappings: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the mapper with KenPom team names.

        Args:
            kenpom_teams: Set of valid KenPom team names
            additional_mappings: Optional extra raw->kenpom mappings
        """
        self.kenpom_teams = kenpom_teams
        self.mappings: dict[str, str] = {}

        # Build lookup dictionaries
        self._kenpom_lower = {t.lower(): t for t in kenpom_teams}
        self._kenpom_no_st = {
            t.replace(" St.", " State").lower(): t for t in kenpom_teams
        }
        self._kenpom_no_state = {
            t.replace(" State", " St.").lower(): t for t in kenpom_teams
        }

        # Apply manual mappings
        for raw_name, kp_name in MANUAL_MAPPINGS.items():
            if kp_name in kenpom_teams:
                self.mappings[raw_name] = kp_name

        # Apply any additional mappings
        if additional_mappings:
            for raw_name, kp_name in additional_mappings.items():
                if kp_name in kenpom_teams:
                    self.mappings[raw_name] = kp_name

    def get_kenpom_name(self, raw_name: str) -> Optional[str]:
        """
        Get the KenPom name for a raw team name.

        Args:
            raw_name: Team name from raw data

        Returns:
            KenPom team name or None if not found
        """
        # Check explicit mapping first
        if raw_name in self.mappings:
            return self.mappings[raw_name]

        # Try exact match
        if raw_name in self.kenpom_teams:
            return raw_name

        raw_lower = raw_name.lower()

        # Remove mascot (last word typically)
        parts = raw_name.split()
        school = " ".join(parts[:-1]).lower() if len(parts) >= 2 else raw_lower

        # Try exact match on school name
        if school in self._kenpom_lower:
            result = self._kenpom_lower[school]
            self.mappings[raw_name] = result  # Cache for future lookups
            return result

        # Try St. <-> State variations
        if school in self._kenpom_no_st:
            result = self._kenpom_no_st[school]
            self.mappings[raw_name] = result
            return result

        if school in self._kenpom_no_state:
            result = self._kenpom_no_state[school]
            self.mappings[raw_name] = result
            return result

        # Try first word match (for unique schools)
        first_word = raw_lower.split()[0]
        matches = [kp for kp in self.kenpom_teams if kp.lower().startswith(first_word)]
        if len(matches) == 1:
            self.mappings[raw_name] = matches[0]
            return matches[0]

        return None

    def build_mapping_for_teams(self, raw_teams: set[str]) -> dict[str, str]:
        """
        Build a complete mapping dictionary for a set of raw team names.

        Args:
            raw_teams: Set of raw team names to map

        Returns:
            Dictionary mapping raw names to KenPom names
        """
        mapping = {}
        for team in raw_teams:
            kp_name = self.get_kenpom_name(team)
            if kp_name:
                mapping[team] = kp_name
        return mapping

    def get_unmapped_teams(self, raw_teams: set[str]) -> set[str]:
        """
        Get teams that couldn't be mapped.

        Args:
            raw_teams: Set of raw team names

        Returns:
            Set of unmapped team names
        """
        unmapped = set()
        for team in raw_teams:
            if self.get_kenpom_name(team) is None:
                unmapped.add(team)
        return unmapped


def build_team_mapping(
    game_teams: set[str],
    kenpom_teams: set[str],
    additional_mappings: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """
    Convenience function to build a team name mapping.

    Args:
        game_teams: Set of team names from game data
        kenpom_teams: Set of team names from KenPom data
        additional_mappings: Optional extra mappings

    Returns:
        Dictionary mapping game team names to KenPom names
    """
    mapper = TeamNameMapper(kenpom_teams, additional_mappings)
    return mapper.build_mapping_for_teams(game_teams)

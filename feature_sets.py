"""@author: Léa Chaccour & Ben Hasenson"""

from enum import Enum

from enum import Enum


class FeatureSet(Enum):
    BASE = "base"  # Base feature set with the most important features
    STRESS_ONLY = "stress_only"  # Focused on stress-related features
    STRESS_FOCUSED_WITH_MILESTONE = (
        "stress_focused_with_milestone"  # Stress features plus milestone
    )
    MILESTONE_FOCUSED = (
        "milestone_focused"  # Focused on milestone and time-based features
    )
    MILESTONE_FOCUSED_WITH_CAVITY = (
        "milestone_focused_with_cavity"  # Milestone plus cavity features
    )
    CAVITY_FOCUSED = "cavity_focused"  # Focused on cavity-related features
    CAVITY_FOCUSED_WITH_STRESS = (
        "cavity_focused_with_stress"  # Cavity features plus stress
    )
    FULL_FEATURE_SET = "full_feature_set"  # Full set with all key features
    REDUCED_FEATURE_SET = (
        "reduced_feature_set"  # Reduced feature set for specific analysis
    )
    COMBINED_FEATURE_SET = "combined_feature_set"  # Combination of important features
    ONLY_ITH0 = "only_ith0"  # Special set with only Ith0 for initial measurements
    ONLY_ITH0_WITH_CAVITY = "only_ith0_with_cavity"
    CAVITY_LENGTH_WITH_ITH_Milestone="FeatureSet.CAVITY_LENGTH_WITH_ITH_Milestone"   
    CAVITY_LENGTH_WITH_ITH="CAVITY_LENGTH_WITH_ITH"

# Feature sets grouped by focus
FEATURE_SETS = {
    # # Base feature set, includes the most important core features
    FeatureSet.BASE: [
        "location_wafer",
        "Milestone",
        "Ith0",
        "Ith48",
        "Ith24",
        "current stress",
    ],
    # # Stress feature sets
    # # These sets focus specifically on stress-related features
    FeatureSet.STRESS_ONLY: [
        "Ith0",
        "Ith48",
        "Ith24",
        "Milestone",
        "current stress",
    ],
    FeatureSet.STRESS_FOCUSED_WITH_MILESTONE: [
        "Ith0",
        "Ith48",
        "Ith24",
        "Current stress factor",
    ],
    # # Milestone-focused feature sets
    # # These sets emphasize the milestone (time-based) relationship
    FeatureSet.MILESTONE_FOCUSED: [
        "Ith0",
        "Ith48",
        "Ith24",
        "Milestone",
    ],
    FeatureSet.MILESTONE_FOCUSED_WITH_CAVITY: [
        "Ith0",
        "Ith48",
        "Ith24",
        "location_wafer",
        "Milestone",
        "ParameterID",
    ],
    # # Cavity-focused feature sets
    # # These sets emphasize cavity length and related features
    FeatureSet.CAVITY_FOCUSED: [
        "Ith0",
        "Ith48",
        "Ith24",
        "ParameterID",
    ],
    FeatureSet.CAVITY_FOCUSED_WITH_STRESS: [
        "Ith0",
        "Ith48",
        "Ith24",
        "ParameterID",
        "Current stress factor",
    ],
    # Full feature set with all key variables
    FeatureSet.FULL_FEATURE_SET: [
        "location_wafer",
        "Milestone",
        "Ith0",
        "Ith48",
        "Ith24",
        "current stress",
        "ParameterID",
    ],
#     # Reduced feature set, useful for specific or targeted analysis
    FeatureSet.REDUCED_FEATURE_SET: [
        "Ith0",
        "Ith48",
        "Ith24",
        "ParameterID",
    ],
#     # Combined feature set, mixing relevant variables for a general view
    FeatureSet.COMBINED_FEATURE_SET: [
        "location_wafer",
        "Ith0",
        "Ith48",
        "Ith24",
        "Current stress factor",
    ],
#     # Special set without ith48 for initial measurements
    FeatureSet.ONLY_ITH0: [
        "location_wafer",
        "Ith0",
        "current stress",
    ],
    FeatureSet.ONLY_ITH0_WITH_CAVITY: [
        "location_wafer",
        "Ith0",
        "ParameterID",
    ],





   FeatureSet.CAVITY_LENGTH_WITH_ITH_Milestone: [
   
    "ParamIDx Ith0",
    "parameterID_x_Ith24",
    "ParamIDx Ith48",
    "current stress",
    "Milestone",
    "location_wafer",
],#set 2 
   FeatureSet.CAVITY_LENGTH_WITH_ITH: [
   
    "ParamIDx Ith0",
    "parameterID_x_Ith24",
    "ParamIDx Ith48",
    "Current stress factor",
    "location_wafer",
],#set 3


}

// UNVALIDATED IN THIS REPOSITORY'S CI: the Android SDK/NDK hosts (dl.google.com) were unreachable in the research container.
// The bridge logic these files depend on is validated separately in ./core-bridge (JVM tests against the real Rust library).
pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositories { google(); mavenCentral() } }
rootProject.name = "aiagent-android"
include(":app")

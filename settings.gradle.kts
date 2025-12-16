pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven {
            url = uri("https://maven.pkg.github.com/facebook/meta-wearables-dat-android")
            credentials {
                username = "" // not needed
                password = System.getenv("GITHUB_TOKEN") ?: try {
                    val localProperties = java.util.Properties()
                    val localPropertiesFile = rootDir.resolve("local.properties")
                    if (localPropertiesFile.exists()) {
                        localProperties.load(localPropertiesFile.inputStream())
                    }
                    localProperties.getProperty("github_token")
                } catch (e: Exception) {
                    null
                }
            }
        }
    }
}

rootProject.name = "Seenitt"
include(":app")

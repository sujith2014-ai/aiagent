plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }

android {
    namespace = "com.aiagent.app"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.aiagent.app"; minSdk = 26; targetSdk = 34; versionCode = 1; versionName = "0.1.0"
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64") }   // the lite core compiles for all three (docs/ANDROID.md); linking/running on devices is unvalidated
    }
    buildTypes { release { isMinifyEnabled = true } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    // Reuse the validated bridge sources unchanged (pure Kotlin, org.json comes from the Android platform).
    sourceSets["main"].kotlin.srcDir("../core-bridge/src/main/kotlin")
}

dependencies { implementation("androidx.core:core-ktx:1.13.1") }

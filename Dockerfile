FROM msranni/nni

    # Install .NET
RUN apt-get update \
    # Install prerequisites
    && apt-get install -y --no-install-recommends \
       wget \
       ca-certificates \
    \
    # Install Microsoft package feed
    && wget -q https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    \
    # Install .NET Core
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        dotnet-sdk-3.1 \
    \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

COPY src/Microsoft.ML.Nni.csproj /app/
RUN dotnet restore /app && rm -rf /app

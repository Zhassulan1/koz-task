#!/usr/bin/env python3

import os
import sys
import json
import subprocess
from typing import Dict, Any
import yaml


def create_opencode_config(ollama_host: str, model_name: str) -> Dict[str, Any]:
    """Create opencode.json configuration"""
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "ollama": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Ollama (local)",
                "options": {
                    "baseURL": ollama_host
                },
                "models": {
                    model_name: {
                        "name": model_name,
                        # "tools": True,
                        "reasoning": True
                    }
                }
            }
        }
    }


def get_user_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default value"""
    if default:
        response = input(f"{prompt} [{default}]: ").strip()
        return response if response else default
    return input(f"{prompt}: ").strip()


def run_command(command: list, cwd: str = None) -> None:
    """Run opencode command and return result"""
    try:
        subprocess.run(
            command,
            cwd=cwd,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout to prevent hanging
            # capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running opencode: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def opencode_installed() -> bool:
    """Check if opencode is installed"""
    try:
        run_command(["opencode", "--version"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def choose_tech_stack() -> Dict[str, str]:
    """Get user's tech stack preferences"""
    print("\nChoose your tech stack:")
    print("1. Full Stack Web (React + Node.js)")
    print("2. Python (FastAPI + React)")
    print("3. Next.js Full Stack")
    print("4. Django + React")
    print("5. Custom (specify your own)")
    
    choice = get_user_input("Select option (1-5)", "1")
    
    stacks = {
        "1": {
            "backend": "Node.js with Express",
            "frontend": "React with TypeScript",
            "description": "Full stack JavaScript application"
        },
        "2": {
            "backend": "Python with FastAPI",
            "frontend": "React with TypeScript",
            "description": "Python backend with React frontend"
        },
        "3": {
            "backend": "Next.js API Routes",
            "frontend": "Next.js with TypeScript",
            "description": "Next.js full stack application"
        },
        "4": {
            "backend": "Django with Django REST Framework",
            "frontend": "React with TypeScript",
            "description": "Django backend with React frontend"
        },
        "5": {
            "backend": get_user_input("Backend technology"),
            "frontend": get_user_input("Frontend technology"),
            "description": "Custom technology stack"
        }
    }
     
    return stacks.get(choice, stacks["1"])


def project_initializer(app_name: str, project_path: str, tech_stack: Dict[str, str], fallback: bool):
    """Use OpenCode to initialize the project structure"""
    print(f"\nInitializing {app_name} with OpenCode...")
    
    # Create the project prompt for opencode
    project_prompt = f"""
        Initialize a new project called "{app_name}" with the following structure and requirements:

        Tech Stack:
        - Backend: {tech_stack['backend']}
        - Frontend: {tech_stack['frontend']}
        - Description: {tech_stack['description']}

        Please create:
        1. A proper folder structure with separate backend and frontend directories
        2. Initialize the backend with appropriate configuration files, dependencies, and basic structure
        3. Initialize the frontend with appropriate configuration files, dependencies, and basic structure
        4. Create a docs folder with basic README.md and project documentation
        5. Add appropriate .gitignore files for each technology
        6. Create a root package.json or requirements.txt as appropriate
        7. Set up basic development scripts and configuration

        Make sure all directories and files are properly initialized with working configurations.
        Start by creating the basic folder structure, then initialize each part properly.
        """

    try:
        # Use opencode to generate the project structure
        print("Running OpenCode to generate project structure...")

        cmd = ["opencode", "run", project_prompt]
        if fallback:
            cmd += ["-m", "opencode/sonic"]

        print(cmd)
        run_command(
            cmd,
            cwd=project_path
        )
        
        print("✓ Project structure created successfully!")
        print("\nProject initialized with the following structure:")

        # Show the created structure
        for root, dirs, files in os.walk(project_path):
            level = root.replace(project_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files to avoid clutter
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")

    except Exception as e:
        print(f"Error initializing project with OpenCode: {e}")
        print("Creating basic folder structure manually...")
        
        # Fallback: create basic structure manually
        os.makedirs(os.path.join(project_path, "backend"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "frontend"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "docs"), exist_ok=True)
        
        # Create basic README
        with open(os.path.join(project_path, "README.md"), "w") as f:
            f.write(f"# {app_name}\n\n{tech_stack['description']}\n\n")
            f.write(f"## Backend\n{tech_stack['backend']}\n\n")
            f.write(f"## Frontend\n{tech_stack['frontend']}\n")


def init_project(app_name: str, path: str, name: str, host: str = "http://localhost:11434/v1"):
    """Initialize a new OpenCode project"""

    # Validate inputs
    if not app_name or not path:
        print("Error: Both app name and path are required")
        sys.exit(1)

    # Check if opencode is installed
    if not opencode_installed():
        print("Error: opencode is not installed or not in PATH")
        print("Please install opencode first: https://opencode.ai/docs/")
        sys.exit(1)

    # Create project directory
    project_path = os.path.abspath(os.path.join(path, app_name))
    os.makedirs(project_path, exist_ok=True)

    print(f"Initializing OpenCode project: {app_name}")
    print(f"Project path: {project_path}")

    # Get Ollama configuration
    print("\nConfigure Ollama settings:")
    fallback_mode = get_user_input("Local models may not be smart enough, fallback to deault opencode sonic (network)? y/n", "y") == "y"
    print(fallback_mode)
    # Create opencode.json
    config = create_opencode_config(host, name)
    config_path = os.path.join(project_path, "opencode.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Created opencode.json at {config_path}")

    # Test opencode configuration
    try:
        print("\nTesting OpenCode configuration...")
        run_command(["opencode", "--version"], cwd=project_path)
        print("✓ OpenCode configuration is valid")
    except Exception as e:
        print(f"Error: OpenCode configuration test failed: {e}")
        sys.exit(1)

    # Get tech stack preferences
    tech_stack = choose_tech_stack()

    # Initialize project with OpenCode
    project_initializer(app_name, project_path, tech_stack, fallback_mode)

    print(f"\n✓ Project {app_name} initialized successfully!")
    print(f"Project location: {project_path}")
    print("\nTo start working on your project:")
    print(f"cd {project_path}")
    print("opencode")


def get_yaml() -> str:
    """Get YAML specification from user input"""
    print("\nPlease paste your YAML module specification:")
    print("(Press Ctrl+D on Unix/Linux/Mac or Ctrl+Z+Enter on Windows when done)")
    
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    
    yaml_content = '\n'.join(lines)
    
    # Validate YAML
    try:
        yaml.safe_load(yaml_content)
        print("✓ YAML specification is valid")
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML specification: {e}")
        sys.exit(1)
    
    return yaml_content


def module_summary(yaml_content: str) -> Dict[str, Any]:
    """Parse and extract information from YAML specification"""
    try:
        spec = yaml.safe_load(yaml_content)
        
        # Extract basic module info
        module_name = spec.get('module', {}).get('name', 'unknown')
        
        # Extract entities info
        entities = spec.get('entities', [])
        entity_summary = []
        for entity in entities:
            fields = entity.get('fields', [])
            entity_summary.append(f"{entity.get('name', 'Unknown')} with {len(fields)} fields")
        
        # Extract API info
        api_info = spec.get('api', {})
        endpoints = api_info.get('endpoints', [])
        
        # Extract UI info
        ui_info = spec.get('ui', {})
        routes = ui_info.get('routes', [])
        
        return {
            'module_name': module_name,
            'entities': entity_summary,
            'api_style': api_info.get('style', 'rest'),
            'base_path': api_info.get('base_path', '/api'),
            'endpoints_count': len(endpoints),
            'ui_framework': ui_info.get('framework', 'react'),
            'routes_count': len(routes),
            'persistence': spec.get('persistence', {}).get('driver', 'postgres')
        }
    except Exception as e:
        print(f"Error parsing YAML specification: {e}")
        sys.exit(1)


def module_generator(module_name: str, yaml_spec: str, spec_info: Dict[str, Any], model_name: str):
    """Use OpenCode to generate module from YAML specification"""
    print(f"\nGenerating module '{module_name}' with OpenCode...")
    
    # Create detailed prompt for OpenCode
    module_prompt = f"""
        Generate a complete module based on this YAML specification:

        {yaml_spec}

        Module Summary:
        - Name: {spec_info['module_name']}
        - Entities: {', '.join(spec_info['entities'])}
        - API Style: {spec_info['api_style']}
        - Base Path: {spec_info['base_path']}
        - Endpoints: {spec_info['endpoints_count']} endpoints
        - UI Framework: {spec_info['ui_framework']}
        - Routes: {spec_info['routes_count']} routes
        - Database: {spec_info['persistence']}

        Please generate:

        1. **Database Layer:**
        - Database migrations for all entities with proper field types
        - Database models/schemas with relationships
        - Repository/DAO patterns for data access

        2. **API Layer:**
        - REST API endpoints as specified
        - Request/response models
        - Proper HTTP status codes and error handling
        - Input validation and sanitization

        3. **Frontend Layer:**
        - UI components for all specified routes
        - Forms for data input with validation
        - Lists/tables for data display
        - Navigation between routes

        4. **Tests:**
        - Unit tests for API endpoints
        - Integration tests for database operations
        - Frontend component tests

        5. **Configuration:**
        - Environment configuration files
        - Database connection setup
        - API route registration
        - Frontend routing setup

        Make sure all code is production-ready with proper error handling, validation, and follows best practices for the specified technologies. Generate working, complete implementations based on the YAML specification.
        """
    
    try:
        print("Running OpenCode to generate module...")
        cmd = ["opencode", "run", module_prompt, "-m", model_name]
        print(cmd)
        run_command(cmd)

        print(f"✓ Module '{module_name}' generated successfully!")
        print("\nModule generation completed with:")
        print("  - Database models and migrations")
        print(f"  - {spec_info['endpoints_count']} API endpoints")
        print(f"  - {spec_info['routes_count']} UI routes")
        print("  - Test files for all components")
        
    except Exception as e:
        print(f"Error generating module with OpenCode: {e}")
        sys.exit(1)


def generate_module(module_name: str, model_name: str):
    """Generate a module from YAML specification"""

    if not module_name:
        print("Error: Module name is required")
        sys.exit(1)

    # Check if opencode is installed
    if not opencode_installed():
        print("Error: opencode is not installed or not in PATH")
        print("Please install opencode first and make sure you're in a configured project directory")
        sys.exit(1)

    # Check if we're in an OpenCode project
    if not os.path.exists("opencode.json"):
        print("Error: Not in an OpenCode project directory")
        print("Please run this command from a directory with opencode.json")
        print("Use 'init <appname> <path>' to create a new project first")
        sys.exit(1)

    print(f"Generating module: {module_name}")

    # Get YAML specification from user
    yaml_spec = get_yaml()

    # Parse and validate specification
    spec_info = module_summary(yaml_spec)

    # Confirm with user
    print("\nModule specification summary:")
    print(f"  Module: {spec_info['module_name']}")
    print(f"  Entities: {len(spec_info['entities'])}")
    print(f"  API Endpoints: {spec_info['endpoints_count']}")
    print(f"  UI Routes: {spec_info['routes_count']}")
    print(f"  Database: {spec_info['persistence']}")

    fallback_mode = get_user_input("Local models may not be smart enough, fallback to deault opencode sonic (network)? y/n", "y") == "y"
    print(fallback_mode)
    if fallback_mode:
        model_name = "opencode/sonic"
    # Generate module with OpenCode
    module_generator(module_name, yaml_spec, spec_info, model_name)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python opencode.py init <project_name> <path>")
        print("  python opencode.py generate_module <module_name>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "init":
        if len(sys.argv) != 4:
            print("Usage: init <project_name> <path>")
            sys.exit(1)

        app_name = sys.argv[2]
        path = sys.argv[3]
        init_project(app_name, path, "placeholder")
    elif command == "generate_module":
        if len(sys.argv) != 3:
            print("Usage: generate_module <module_name>")
            sys.exit(1)

        module_name = sys.argv[2]
        generate_module(module_name, "placeholder")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: init")
        sys.exit(1)


if __name__ == "__main__":
    main()

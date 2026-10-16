# Database Connectivity Troubleshooting Runbook

## Overview

Database connectivity problems occur when an application cannot establish or maintain a connection to its database.

In Kubernetes environments, these failures may cause application startup errors, failed readiness checks, request timeouts, or repeated container restarts.

Common causes include incorrect connection settings, unavailable database services, DNS failures, authentication errors, network restrictions, and database connection limits.

## Symptoms

- Application logs contain connection timeout or connection refused errors.
- The application reports authentication failures when connecting to the database.
- A pod starts but cannot become ready.
- A container repeatedly restarts after failing to initialize its database connection.
- Database queries fail intermittently.
- The application cannot resolve the database hostname.
- Connections fail after a deployment or configuration change.

## 1. Inspect the Application Pod

Identify the affected pod and its namespace.

Run:

    kubectl get pods -n <namespace>

Inspect the pod's events and container states:

    kubectl describe pod <pod-name> -n <namespace>

Review the application logs:

    kubectl logs <pod-name> -n <namespace>

If the container has restarted, inspect the previous container's logs:

    kubectl logs <pod-name> -n <namespace> --previous

Look for database connection errors, DNS failures, authentication errors, and connection timeouts.

## 2. Verify Database Configuration

Inspect the application's Deployment:

    kubectl get deployment <deployment-name> -n <namespace> -o yaml

Verify that:

- The database hostname is correct.
- The database port matches the database listener.
- The database name is correct.
- Required environment variables are present.
- ConfigMap and Secret references are correct.
- Configuration changes have reached the running workload.

Do not print or share database passwords, tokens, or other secret values.

## 3. Check the Database Service

If the database runs inside Kubernetes, list Services:

    kubectl get services -n <database-namespace>

Describe the relevant Service:

    kubectl describe service <database-service> -n <database-namespace>

Inspect its EndpointSlices:

    kubectl get endpointslices -n <database-namespace>

Check the database pods:

    kubectl get pods -n <database-namespace>

Verify that the Service name and port match the application's configuration and that healthy database pods are available.

If the database is hosted outside Kubernetes, verify its hostname, network reachability, listener port, and availability using the appropriate tools.

## 4. Diagnose DNS Resolution

If the application cannot resolve the database hostname, check the hostname, namespace, and cluster DNS health.

Inspect the cluster DNS pods:

    kubectl get pods -n kube-system

Where diagnostic tools are available in an authorized troubleshooting container, test name resolution:

    nslookup <database-hostname>

A Kubernetes Service can commonly be addressed by its name within the same namespace. Cross-namespace access generally requires a namespace-qualified Service name.

Do not assume DNS is the problem solely because a connection timed out. Confirm the failure with diagnostic evidence.

## 5. Test Network Connectivity

If an appropriate diagnostic tool is available, test TCP connectivity:

    nc -vz <database-hostname> <database-port>

Interpret the result:

- Connection succeeds: basic TCP connectivity is available, but authentication and database operations may still fail.
- Connection refused: the destination may be reachable, but no process is accepting connections on that port.
- Connection times out: investigate routing, firewall rules, NetworkPolicies, destination availability, and other network restrictions.
- Hostname resolution fails: investigate DNS and the configured hostname.

Inspect NetworkPolicies:

    kubectl get networkpolicies -n <namespace>

Check whether the policies permit the required application-to-database traffic.

Do not disable network security controls as a troubleshooting shortcut.

## 6. Verify Credentials and Authentication

If the database reports authentication failures, verify that the application uses the correct credentials.

Inspect Secret metadata and references without exposing secret values:

    kubectl describe secret <secret-name> -n <namespace>

Confirm that:

- The Secret exists in the expected namespace.
- The application references the correct Secret and key names.
- The credentials are current.
- The database account has the required permissions.
- Credential rotation has been applied consistently.

Follow the approved deployment process to refresh configuration when required.

## 7. Investigate Timeouts and Connection Limits

Connection timeouts can result from network problems, an overloaded database, slow initialization, or an application timeout that is too aggressive.

Investigate:

- Database CPU and memory pressure.
- Active connections and configured connection limits.
- Long-running queries and database locks.
- Application connection pool settings.
- Application startup timeout settings.
- Recent traffic increases or deployment changes.

Increasing timeouts will not fix incorrect hostnames, refused connections, invalid credentials, or blocked network traffic.

Avoid increasing connection pool sizes without checking database capacity.

## 8. Apply Remediation Carefully

Choose a corrective action based on the evidence.

Examples include:

- Correcting an invalid database hostname or port.
- Restoring database availability.
- Fixing a Service selector or missing endpoints.
- Correcting DNS configuration.
- Updating an incorrect Secret reference.
- Rotating invalid credentials through the approved process.
- Correcting a restrictive NetworkPolicy.
- Adjusting connection pool settings based on database capacity.
- Addressing database resource pressure or excessive query latency.

Make changes through the repository and deployment workflow used by your environment.

Avoid exposing secrets or making production changes without assessing their impact.

## 9. Verify Recovery

After remediation:

1. Confirm that the database is healthy and accepting connections.
2. Verify that the application can resolve the database hostname.
3. Confirm that the required network path is available.
4. Check application logs for successful database initialization or connection recovery.
5. Confirm that the application pod becomes Ready.
6. Monitor application errors, latency, and restart counts.
7. Verify that the failure does not recur under representative workload.

Useful commands:

    kubectl get pods -n <namespace>

    kubectl get events -n <namespace> --sort-by=.lastTimestamp

    kubectl logs <pod-name> -n <namespace> --tail=100

Successful TCP connectivity alone does not prove that authentication, authorization, database queries, and application operations are working correctly.

## Troubleshooting Notes

- Connection timeouts and connection refused errors indicate different failure conditions.
- Investigate DNS resolution, TCP connectivity, authentication, and database queries as separate stages.
- Kubernetes Services, NetworkPolicies, Secrets, and application environment variables are namespace-sensitive where applicable.
- The absence of recent database connection errors does not prove that the problem is resolved.
- When evidence is insufficient, collect more diagnostic information before selecting a remediation.

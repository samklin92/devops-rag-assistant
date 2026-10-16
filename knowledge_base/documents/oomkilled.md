# OOMKilled: Kubernetes Container Memory Troubleshooting

## Overview

OOMKilled is a container termination reason indicating that the container was killed during an out-of-memory event. In Kubernetes, this commonly occurs when a container exceeds its configured memory limit.

Exit code 137 commonly indicates that a process received SIGKILL (signal 9). It is not conclusive evidence of an out-of-memory event by itself. Check the container termination reason and relevant events to establish the cause.

## Symptoms

* `kubectl describe pod` shows a container's last termination reason as `OOMKilled`.
* A container may show exit code 137.
* A pod may repeatedly restart if the application is killed and its restart policy causes it to restart.
* Container memory usage may approach its configured limit before termination.
* Application logs may show incomplete processing because the process was terminated abruptly.

## Initial Investigation

### 1. Inspect the pod

Run:

```bash
kubectl get pods -n <namespace>
```

Check the pod's status and restart count.

### 2. Inspect termination details

Run:

```bash
kubectl describe pod <pod-name> -n <namespace>
```

Look for:

* Last State
* Reason
* Exit Code
* Restart Count
* Events

A termination reason of `OOMKilled` supports the diagnosis of a container memory-related termination.

### 3. Inspect previous container logs

Run:

```bash
kubectl logs <pod-name> -n <namespace> --previous
```

Previous logs may reveal memory-intensive operations, unusually large requests, or workload patterns preceding termination. Logs may be incomplete if the process was killed abruptly.

### 4. Inspect resource configuration

Run:

```bash
kubectl get pod <pod-name> -n <namespace> -o yaml
```

Review the affected container's resource requests and limits.

Example:

```yaml
resources:
  requests:
    memory: "128Mi"
  limits:
    memory: "256Mi"
```

A memory limit constrains how much memory the container can use. A memory request is used by Kubernetes for scheduling and does not impose the same hard upper bound.

## Common Causes

* The container's memory limit is too low for its workload.
* The application has a memory leak.
* Requests cause unusually high memory consumption.
* The application loads large datasets or files into memory.
* Concurrent workloads increase memory demand.
* A node experiences memory pressure, resulting in an out-of-memory event.

## Remediation

### 1. Determine whether the limit is appropriate

Compare observed memory usage with the configured limit and the application's normal workload requirements.

If the limit is too low, adjust it based on measured usage and available cluster capacity.

### 2. Investigate application memory usage

Review application metrics, profiles, logs, and workload patterns to identify excessive memory consumption.

If memory usage grows continuously over time, investigate a possible memory leak.

### 3. Check current resource usage

If Metrics Server is available, run:

```bash
kubectl top pod <pod-name> -n <namespace>
```

This command reports current usage; it does not reconstruct historical usage at the time of a previous crash.

### 4. Check node conditions

Run:

```bash
kubectl describe node <node-name>
```

Inspect node conditions and events for memory pressure or related resource problems.

### 5. Apply changes carefully

Update resource limits or application behavior through the repository and deployment workflow used by your environment. Avoid increasing memory limits without checking node capacity and workload requirements.

## Verification

After remediation:

1. Confirm the pod becomes healthy.
2. Monitor its restart count.
3. Observe memory usage under representative workloads.
4. Check whether new OOMKilled terminations occur.
5. Review application metrics and logs for recurring memory problems.

Useful commands:

```bash
kubectl get pods -n <namespace>
kubectl describe pod <pod-name> -n <namespace>
kubectl top pod <pod-name> -n <namespace>
kubectl get events -n <namespace> --sort-by=.lastTimestamp
```

## Troubleshooting Notes

* Exit code 137 alone does not establish that a container was OOMKilled.
* A pod can have memory-related problems even when a single usage snapshot looks normal.
* Increasing memory limits without investigating the cause may hide an application memory leak.
* A container termination reason of `OOMKilled` is stronger evidence than exit code 137 alone.
* If the termination reason is not `OOMKilled`, investigate other possible causes of SIGKILL or process termination.

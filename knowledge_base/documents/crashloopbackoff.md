# CrashLoopBackOff Troubleshooting

## Problem Description

CrashLoopBackOff is a Kubernetes pod state. It means a container in the pod starts, exits, and is restarted repeatedly. Kubernetes increases the delay between restarts each time, up to a maximum of five minutes. CrashLoopBackOff is a symptom, not a root cause: the real problem is whatever makes the container exit.

## Symptoms for CrashLoopBackOff

- `kubectl get pods` shows the status `CrashLoopBackOff`.
- The RESTARTS column keeps increasing.
- The pod alternates between `Running`, `Error` and `CrashLoopBackOff`.

## Possible Causes of CrashLoopBackOff

- The application crashes on startup because of a bug or an unhandled exception.
- A required environment variable, ConfigMap or Secret is missing or incorrect.
- The container cannot reach a dependency at startup, such as a database, and exits when the connection times out.
- A liveness probe fails, so Kubernetes kills and restarts the container.
- The container exceeds its memory limit and is terminated (OOMKilled).
- The container command or entrypoint is wrong, so the process exits immediately.
- A file or volume the application needs is missing or has wrong permissions.

## Diagnostic Commands for CrashLoopBackOff

List pods and their restart counts:

    kubectl get pods -n <namespace>

Show the pod's events, last state and exit code:

    kubectl describe pod <pod-name> -n <namespace>

Read the logs of the current container:

    kubectl logs <pod-name> -n <namespace>

Read the logs of the previous, crashed container. This is usually the most useful command:

    kubectl logs <pod-name> -n <namespace> --previous

Show recent events in the namespace, newest last:

    kubectl get events -n <namespace> --sort-by=.lastTimestamp

## Troubleshooting Steps for CrashLoopBackOff

1. Run `kubectl describe pod` and find the `Last State` section. Note the `Reason` and the `Exit Code`.
2. Interpret the exit code. Exit code 1 usually means an application error. Exit code 137 means the process was killed with SIGKILL, often because of an out-of-memory condition. Exit code 143 means it received SIGTERM.
3. Read the previous container logs with `kubectl logs --previous` and look for the first error before the exit.
4. If the logs show a connection failure or timeout, check the dependency the application connects to. See the database connectivity runbook.
5. If the `Reason` is `OOMKilled`, see the OOMKilled runbook.
6. Check the `Events` section of `kubectl describe pod` for failed probes. If a liveness probe is failing, compare the probe settings to how long the application actually takes to start.
7. Verify that every ConfigMap, Secret and environment variable the pod references exists in the namespace.
8. Check whether the issue began after a deployment by reviewing the rollout history.

## Remediation for CrashLoopBackOff

- Fix the configuration error, then let Kubernetes restart the pod with the corrected settings.
- If a liveness probe is too aggressive, increase `initialDelaySeconds` or `failureThreshold`, or add a startup probe.
- If memory is the cause, raise the memory limit or fix the memory leak.
- If a recent deployment introduced the failure, roll back to the previous revision. See the application rollback runbook.

## Warnings for CrashLoopBackOff

- Deleting the pod does not fix the root cause. The replacement pod will crash the same way.
- Changing resource limits or probe settings in production restarts pods and can reduce capacity. Make the change during a safe window.
- Do not edit a running production deployment by hand without recording the change. Manual changes are overwritten by the next deployment.
- Logs can contain secrets or personal data. Do not paste them into public tickets or chat tools.

## Related Runbooks

- database-connectivity.md
- oomkilled.md
- kubernetes-readiness-probes.md
- application-rollback.md

## Troubleshooting Missing ConfigMaps and Secrets

Use this section when a Kubernetes application fails to start because a referenced ConfigMap, Secret, environment variable, or configuration key is missing or incorrect.

### Check for Missing Configuration References

Identify the affected pod and namespace:

    kubectl describe pod <pod-name> -n <namespace>

Inspect the Events section for messages indicating that a ConfigMap or Secret was not found, a required key is missing, or a volume could not be mounted.

List ConfigMaps and Secrets in the application's namespace:

    kubectl get configmaps -n <namespace>
    kubectl get secrets -n <namespace>

Verify that each ConfigMap and Secret referenced by the Deployment exists in the same namespace as the workload.

Inspect the Deployment configuration:

    kubectl get deployment <deployment-name> -n <namespace> -o yaml

Check `env`, `envFrom`, `configMapKeyRef`, `secretKeyRef`, and configuration volume references. Confirm that referenced object names and required key names match the actual ConfigMaps and Secrets.

For environment variables marked as required, verify that the application receives the expected variable names. Do not print, expose, or share secret values.

### Remediate Missing Configuration

- Create or restore the missing ConfigMap or Secret through the approved deployment workflow.
- Correct invalid object names, key names, or namespace references.
- Ensure that required configuration keys exist.
- Redeploy or restart the workload when necessary for updated configuration to take effect.
- Verify that the pod starts successfully and inspect its logs for remaining errors.

If the pod continues restarting after configuration is corrected, review its current logs and previous container logs to identify the next failure.

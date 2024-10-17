import grp
import json
import logging
import os
import pwd
import subprocess
from typing import List, Optional

import fire
import yaml
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RichHandler(markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

MAX_CPU = 192
MAX_RAM = 890
MAX_GPU = 8


def fetch_user_info():
    user_info = {}

    # Get the current user name
    user_info["login_user"] = os.getlogin()

    # Get user entry from /etc/passwd
    pw_entry = pwd.getpwnam(os.getlogin())

    # Extracting home directory and shell from the password entry
    user_info["home"] = pw_entry.pw_dir
    user_info["shell"] = pw_entry.pw_shell

    # Get group IDs
    group_ids = os.getgrouplist(os.getlogin(), pw_entry.pw_gid)

    # Get group names from group IDs
    user_info["groups"] = " ".join(
        [grp.getgrgid(gid).gr_name for gid in group_ids]
    )

    return user_info


class GPU_TYPE:
    NVIDIA_A100 = "NVIDIA-A100"
    NVIDIA_L4 = "NVIDIA-L4"


class GPU_MEMORY:
    GPU_40GB = "40GB"
    GPU_80GB = "80GB"
    GPU_24GB = "24GB"


class GPU_NUMBER:
    GPU_1 = 1
    GPU_2 = 2
    GPU_4 = 4
    GPU_8 = 8


class KubernetesJob:
    """
    A class for generating Kubernetes Job YAML configurations.

    Attributes:
        name (str): Name of the job and associated resources.
        image (str): Container image to use for the job.
        command (List[str], optional): Command to execute in the container. Defaults to None.
        args (List[str], optional): Arguments for the command. Defaults to None.
        storage_request (str, optional): Amount of storage to request. For example, "10Gi" for 10 gibibytes. Defaults to None.
        gpu_type (str, optional): Type of GPU resource, e.g. "nvidia.com/gpu". Defaults to None.
        gpu_number (int, optional): Number of GPU resources to allocate. Defaults to None.
        backoff_limit (int, optional): Maximum number of retries before marking job as failed. Defaults to 4.
        restart_policy (str, optional): Restart policy for the job, default is "Never".
        shm_size (str, optional): Size of shared memory, e.g. "2Gi". If not set, defaults to None.
        secret_env_vars (dict, optional): Dictionary of secret environment variables. Defaults to None.
        env_vars (dict, optional): Dictionary of normal (non-secret) environment variables. Defaults to None.
        volume_mounts (dict, optional): Dictionary of volume mounts. Defaults to None.
        namespace (str, optional): Namespace of the job. Defaults to None.

    Methods:
        generate_yaml() -> dict: Generate the Kubernetes Job YAML configuration.
    """

    def __init__(
        self,
        name: str,
        image: str,
        command: List[str] = None,
        args: Optional[List[str]] = None,
        storage_request: Optional[str] = None,
        gpu_type: Optional[str] = None,
        gpu_number: Optional[int] = None,
        gpu_memory: Optional[str] = None,
        backoff_limit: int = 0,
        restart_policy: str = "Never",
        shm_size: Optional[str] = None,
        secret_env_vars: Optional[dict] = None,
        env_vars: Optional[dict] = None,
        volume_mounts: Optional[dict] = None,
        job_deadlineseconds: Optional[int] = None,
        privileged_security_context: bool = False,
        user_name: Optional[str] = None,
        user_email: Optional[str] = None,
        labels: Optional[dict] = None,
        annotations: Optional[dict] = None,
        namespace: Optional[str] = None,
        image_pull_secret: Optional[str] = None,
        node_selector: Optional[dict] = None,  # 🆕 Added node_selector
        tolerations: Optional[List[dict]] = None,  # 🆕 Added tolerations
        resources: Optional[dict] = None,  # 🆕 Added resources
    ):
        self.name = name

        self.image = image
        self.command = command
        self.args = args
        self.storage_request = storage_request
        self.gpu_type = gpu_type

        self.backoff_limit = backoff_limit
        self.restart_policy = restart_policy

        if isinstance(shm_size, int):
            shm_size = f"{shm_size}G"

        self.shm_size = (
            shm_size if shm_size is not None else f"{80 * int(gpu_number)}G"
        )
        self.secret_env_vars = secret_env_vars
        self.image_pull_secret = image_pull_secret
        self.env_vars = env_vars
        self.volume_mounts = volume_mounts
        self.job_deadlineseconds = job_deadlineseconds
        self.privileged_security_context = privileged_security_context

        self.user_name = user_name or os.environ.get("USER", "unknown")
        self.user_email = user_email  # This is now a required field.

        self.gpu_memory = gpu_memory
        self.gpu_number = gpu_number
        self.gpu_type = gpu_type

        self.node_selector = node_selector or {}
        self.tolerations = tolerations or []
        self.resources = resources or {}

        # Update labels with GPU-specific information
        self.labels = {
            "job/user": self.user_name,
        }
        if gpu_type:
            self.labels["nvidia.com/gpu.type"] = gpu_type
        if gpu_number:
            self.labels["nvidia.com/gpu.number"] = str(gpu_number)
        if gpu_memory:
            self.labels["nvidia.com/gpu.memory"] = gpu_memory

        if labels is not None:
            self.labels.update(labels)

        self.annotations = {"job/user": self.user_name}
        if user_email is not None:
            self.annotations["job/email"] = user_email

        if annotations is not None:
            self.annotations.update(annotations)

        self.user_info = fetch_user_info()
        self.annotations.update(self.user_info)
        logger.info(f"labels {self.labels}")
        logger.info(f"annotations {self.annotations}")

        self.namespace = namespace

    def _setup_node_selector(self) -> dict:
        """Set up node selector based on GPU requirements and user-provided selectors."""
        node_selector = self.node_selector.copy()
        if self.gpu_type:
            node_selector.setdefault("nvidia.com/gpu.type", self.gpu_type)
        if self.gpu_memory:
            node_selector.setdefault("nvidia.com/gpu.memory", self.gpu_memory)
        if self.gpu_number:
            node_selector.setdefault(
                "nvidia.com/gpu.number", str(self.gpu_number)
            )
        return node_selector

    def _add_shm_size(self, container: dict):
        """Adds shared memory volume if shm_size is set."""
        if self.shm_size:
            container["volumeMounts"].append(
                {"name": "dshm", "mountPath": "/dev/shm"}
            )
        return container

    def _add_env_vars(self, container: dict):
        """Adds secret and normal environment variables to the
        container."""
        container["env"] = []
        if self.secret_env_vars or self.env_vars:
            if self.secret_env_vars:
                for key, value in self.secret_env_vars.items():
                    container["env"].append(
                        {
                            "name": key,
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": value["secret_name"],
                                    "key": value["key"],
                                }
                            },
                        }
                    )

            if self.env_vars:
                for key, value in self.env_vars.items():
                    container["env"].append({"name": key, "value": value})

        # Always export the POD_NAME environment variable
        container["env"].append(
            {
                "name": "POD_NAME",
                "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
            }
        )

        return container

    def _add_volume_mounts(self, container: dict):
        """Adds volume mounts to the container."""
        if self.volume_mounts:
            for mount_name, mount_data in self.volume_mounts.items():
                container["volumeMounts"].append(
                    {
                        "name": mount_name,
                        "mountPath": mount_data["mountPath"],
                    }
                )

        return container

    def _add_privileged_security_context(self, container: dict):
        """Adds privileged security context to the container."""
        if self.privileged_security_context:
            container["securityContext"] = {
                "privileged": True,
            }

        return container

    def generate_yaml(self):
        container = {
            "name": self.name,
            "image": self.image,
            "imagePullPolicy": "Always",
            "volumeMounts": [],
            "resources": self.resources
            or {
                "requests": {},
                "limits": {},
            },
        }

        if self.command is not None:
            container["command"] = self.command

        if self.args is not None:
            container["args"] = self.args

        container = self._add_shm_size(container)
        container = self._add_env_vars(container)
        container = self._add_volume_mounts(container)
        container = self._add_privileged_security_context(container)

        if self.storage_request is not None:
            container["resources"]["requests"][
                "storage"
            ] = self.storage_request

        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": self.name,
                "labels": self.labels,
                "annotations": self.annotations,
            },
            "spec": {
                "template": {
                    "metadata": {
                        "labels": self.labels,
                        "annotations": self.annotations,
                    },
                    "spec": {
                        "containers": [container],
                        "restartPolicy": self.restart_policy,
                        "volumes": [],
                    },
                },
                "backoffLimit": self.backoff_limit,
            },
        }

        # Add node selector
        node_selector = self._setup_node_selector()
        if node_selector:
            job["spec"]["template"]["spec"]["nodeSelector"] = node_selector

        # Add tolerations
        if self.tolerations:
            job["spec"]["template"]["spec"]["tolerations"] = self.tolerations

        # Add GPU resources if specified
        if self.gpu_number and not self.resources:
            container["resources"]["limits"][
                "nvidia.com/gpu"
            ] = self.gpu_number
            container["resources"]["requests"][
                "nvidia.com/gpu"
            ] = self.gpu_number

        if self.job_deadlineseconds:
            job["spec"]["activeDeadlineSeconds"] = self.job_deadlineseconds

        if self.namespace:
            job["metadata"]["namespace"] = self.namespace

        # Add shared memory volume if shm_size is set
        if self.shm_size:
            job["spec"]["template"]["spec"]["volumes"].append(
                {
                    "name": "dshm",
                    "emptyDir": {
                        "medium": "Memory",
                        "sizeLimit": self.shm_size,
                    },
                }
            )

        # Add volumes for the volume mounts
        if self.volume_mounts:
            for mount_name, mount_data in self.volume_mounts.items():
                volume = {"name": mount_name}

                if "pvc" in mount_data:
                    volume["persistentVolumeClaim"] = {
                        "claimName": mount_data["pvc"]
                    }
                elif "emptyDir" in mount_data:
                    volume["emptyDir"] = {}
                # Add more volume types here if needed
                if "server" in mount_data:
                    volume["nfs"] = {
                        "server": mount_data["server"],
                        "path": mount_data["path"],
                    }

                job["spec"]["template"]["spec"]["volumes"].append(volume)

        if self.image_pull_secret:
            job["spec"]["template"]["spec"]["imagePullSecrets"] = [
                {"name": self.image_pull_secret}
            ]

        return yaml.dump(job)

    def run(self):
        from kubernetes import config

        config.load_kube_config()
        job_yaml = self.generate_yaml()

        # Save the generated YAML to a temporary file
        with open("temp_job.yaml", "w") as temp_file:
            temp_file.write(job_yaml)

        # Run the kubectl command with --validate=False
        cmd = ["kubectl", "apply", "-f", "temp_job.yaml"]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Remove the temporary file
            os.remove("temp_job.yaml")
            return result.returncode
        except subprocess.CalledProcessError as e:
            logger.info(
                f"Command '{' '.join(cmd)}' failed with return code {e.returncode}."
            )
            logger.info(f"Stdout:\n{e.stdout}")
            logger.info(f"Stderr:\n{e.stderr}")
            # Remove the temporary file
            os.remove("temp_job.yaml")
            return e.returncode  # return the exit code
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred while running '{' '.join(cmd)}'."
            )  # This logs the traceback too
            # Remove the temporary file
            os.remove("temp_job.yaml")
            return 1  # return the exit code

    @classmethod
    def from_command_line(cls):
        """Create a KubernetesJob instance from command-line arguments
        and run the job.
        Example: python kubejobs/jobs.py --image=nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 --gpu_type=nvidia.com/gpu --gpu_limit=1 --backoff_limit=4 --gpu_product=NVIDIA-A100-SXM4-40GB
        """
        fire.Fire(cls)


def create_jobs_for_experiments(commands: List[str], *args, **kwargs):
    """
    Creates and runs a Kubernetes Job for each command in the given list of commands.

    :param commands: A list of strings, where each string represents a command to be executed.
    :param args: Positional arguments to be passed to the KubernetesJob constructor.
    :param kwargs: Keyword arguments to be passed to the KubernetesJob constructor.

    :Example:

    .. code-block:: python

        from kubejobs import KubernetesJob

        commands = [
            "python experiment.py --param1 value1",
            "python experiment.py --param1 value2",
            "python experiment.py --param1 value3"
        ]

        create_jobs_for_experiments(
            commands,
            image="nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04",
            gpu_type="nvidia.com/gpu",
            gpu_limit=1,
            backoff_limit=4
        )
    """
    jobs = []
    for idx, command in enumerate(commands):
        job_name = f"{kwargs.get('name', 'experiment')}-{idx}"
        kubernetes_job = KubernetesJob(
            name=job_name,
            command=["/bin/bash"],
            args=["-c", command],
            *args,
            **kwargs,
        )
        kubernetes_job.run()
        jobs.append(kubernetes_job)

    return jobs


def create_pvc(
    pvc_name: str,
    storage: str,
    access_modes: list = None,
):
    if access_modes is None:
        access_modes = ["ReadWriteOnce"]

    if isinstance(access_modes, str):
        access_modes = [access_modes]

    pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": pvc_name},
        "spec": {
            "accessModes": access_modes,
            "resources": {"requests": {"storage": storage}},
        },
    }

    # Convert the PVC dictionary to a JSON string
    pvc_json = json.dumps(pvc)

    # Write the JSON to a temporary file
    with open("pvc.json", "w") as f:
        f.write(pvc_json)

    # Use kubectl to create the PVC from the JSON file
    subprocess.run(["kubectl", "apply", "-f", "pvc.json"], check=True)

    # Clean up the temporary file
    subprocess.run(["rm", "pvc.json"], check=True)

    return pvc_name


def create_pv(
    pv_name: str,
    storage: str,
    storage_class_name: str,
    access_modes: List[str],
    pv_type: str,
    namespace: str = "default",
    claim_name: Optional[str] = None,
    local_path: Optional[str] = None,
    fs_type: str = "ext4",
) -> None:
    """
    Create a PersistentVolume using kubectl commands.

    :param pv_name: The name of the PersistentVolume.
    :param storage: The amount of storage for the PersistentVolume (e.g., "1500Gi").
    :param storage_class_name: The storage class name for the PersistentVolume.
    :param access_modes: A list of access modes for the PersistentVolume.
    :param pv_type: The type of PersistentVolume, either 'local' or 'node'.
    :param namespace: The namespace in which to create the PersistentVolume. Defaults to "default".
    :param claim_name: The name of the PersistentVolumeClaim to bind to the PersistentVolume.
    :param local_path: The path on the host for a local PersistentVolume. Required if pv_type is 'local'.
    :param fs_type: The filesystem type for the PersistentVolume. Defaults to "ext4".

    :raises ValueError: If pv_type is not 'local' or 'node', or if local_path is not provided for 'local' type.
    :raises subprocess.CalledProcessError: If the kubectl command fails.
    """
    if pv_type not in ["local", "node"]:
        raise ValueError("pv_type must be either 'local' or 'node'")

    if pv_type == "local" and not local_path:
        raise ValueError("local_path must be provided when pv_type is 'local'")

    pv = {
        "apiVersion": "v1",
        "kind": "PersistentVolume",
        "metadata": {"name": pv_name},
        "spec": {
            "storageClassName": storage_class_name,
            "capacity": {"storage": storage},
            "accessModes": access_modes,
            "csi": {"driver": "pd.csi.storage.gke.io", "fsType": fs_type},
        },
    }

    if claim_name:
        pv["spec"]["claimRef"] = {"namespace": namespace, "name": claim_name}

    if pv_type == "local":
        pv["spec"]["hostPath"] = {"path": local_path}

    # Convert the PV dictionary to a JSON string
    pv_json = json.dumps(pv)

    # Write the JSON to a temporary file
    with open("pv.json", "w") as f:
        f.write(pv_json)

    # Use kubectl to create the PV from the JSON file
    try:
        subprocess.run(
            ["kubectl", "apply", "-f", "pv.json"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Successfully created PersistentVolume: {pv_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create PersistentVolume: {pv_name}")
        logger.error(f"Error: {e.stderr}")
        raise
    finally:
        # Clean up the temporary file
        subprocess.run(["rm", "pv.json"], check=True)


if __name__ == "__main__":
    KubernetesJob.from_command_line()

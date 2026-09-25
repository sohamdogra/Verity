import { Suspense } from "react";
import { Loader2 } from "lucide-react";
import { AlertConfirmation } from "./alert-confirmation";

export default function AlertPage() {
  return (
    <Suspense fallback={<Loader2 className="mx-auto mt-16 size-10 animate-spin text-primary" />}>
      <AlertConfirmation />
    </Suspense>
  );
}
